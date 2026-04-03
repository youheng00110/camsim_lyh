import bisect
import math
import numpy as np
from PIL import ImageDraw
import torch
import transforms3d
import torch.nn.functional as F
import random
import os, time
import cv2

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import InterpolationMode

INVALID_BIN = 0  # proj_depth 是 bins_u16 时，无效值是 0


def rs_bins_u16(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    x:
      - depth bins: [T,V,H,W] long/int
      - sem masks : [T,V,C,H,W] long/int/float (一般 0/1)
    return:
      - same ndim, resized to target H,W
    resize rule: nearest (不会引入新类别)
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"[rs_bins_u16] invalid target size: H={height}, W={width}")

    if not torch.is_tensor(x):
        raise TypeError(f"[rs_bins_u16] expect torch.Tensor, got {type(x)}")

    # case A: [T,V,H,W]
    if x.ndim == 4:
        T, V, H0, W0 = x.shape
        y = F.interpolate(
            x.reshape(T * V, 1, H0, W0).float(),
            size=(height, width),
            mode="nearest"
        )
        return y.reshape(T, V, height, width).long()

    # case B: [T,V,C,H,W]
    if x.ndim == 5:
        T, V, C, H0, W0 = x.shape
        y = F.interpolate(
            x.reshape(T * V, C, H0, W0).float(),
            size=(height, width),
            mode="nearest"
        )
        # sem 通常你希望保持 float(0/1)，也可以 .long()
        return y.reshape(T, V, C, height, width)

    raise ValueError(f"[rs_bins_u16] expect [T,V,H,W] or [T,V,C,H,W], got {tuple(x.shape)}")

def resize_clr_keep_invalid(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    x: torch.Tensor [T,V,3,H,W]  float in [0,1] (推荐) 或 uint8
    return: torch.FloatTensor [T,V,3,height,width] in [0,1]
    规则：
      - 下采样：黑色像素(0,0,0)不参与平均（weighted area）
      - 上采样：nearest
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] invalid target size: H={height}, W={width}")

    if not torch.is_tensor(x):
        raise TypeError(f"[resize_clr_keep_invalid_tvchw] expect torch.Tensor, got {type(x)}")

    if x.ndim != 5:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] expect [T,V,3,H,W], got {tuple(x.shape)}")

    T, V, C, H0, W0 = x.shape
    if C != 3:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] channel must be 3, got C={C}")

    # to float [0,1]
    if x.dtype == torch.uint8:
        x_f = x.float() / 255.0
    else:
        x_f = x.float()

    x_f = x_f.reshape(T * V, 3, H0, W0)

    # upsample -> nearest
    if height >= H0 and width >= W0:
        y = F.interpolate(x_f, size=(height, width), mode="nearest")
        return y.reshape(T, V, 3, height, width)

    # downsample -> weighted area (ignore black)
    valid = (x_f.sum(dim=1, keepdim=True) > 0).float()  # [TV,1,H,W]

    num = F.interpolate(x_f * valid, size=(height, width), mode="area")
    den = F.interpolate(valid,       size=(height, width), mode="area")

    y = torch.zeros_like(num)
    m = den > 1e-6
    y[m.expand_as(y)] = (num / den.clamp_min(1e-6))[m.expand_as(y)]

    return y.reshape(T, V, 3, height, width)


class Copy():
    def __call__(self, a):
        return a


class FilterPoints():
    def __init__(self, min_distance: float = 0, max_distance: float = 1000.0):
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __call__(self, a):
        distances = a[:, :3].norm(dim=-1)
        mask = torch.logical_and(
            distances >= self.min_distance, distances < self.max_distance)

        return a[mask]


class TakePoints():
    def __init__(self, max_count: int = 32768):
        self.max_count = max_count

    def __call__(self, a):
        if a.shape[0] > self.max_count:
            indices = torch.randperm(a.shape[0])[:self.max_count]
            a = a[indices]

        return a

def resize_and_pad_image(img, target_h, target_w, pad_value=0.0):
    x = to_chw_tensor(img)
    c, h0, w0 = x.shape

    scale = min(float(target_w) / float(w0), float(target_h) / float(h0))
    hs = max(1, int(round(h0 * scale)))
    ws = max(1, int(round(w0 * scale)))

    x = F.interpolate(
        x.unsqueeze(0),
        size=(hs, ws),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    pad_top = (target_h - hs) // 2
    pad_bottom = target_h - hs - pad_top
    pad_left = (target_w - ws) // 2
    pad_right = target_w - ws - pad_left

    y = F.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=pad_value
    )

    meta = {
        "orig_h": h0,
        "orig_w": w0,
        "scaled_h": hs,
        "scaled_w": ws,
        "new_h": target_h,
        "new_w": target_w,
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    return y, meta


def make_valid_mask_from_meta(meta):
    mask = torch.zeros(1, meta["new_h"], meta["new_w"], dtype=torch.float32)
    top = meta["pad_top"]
    left = meta["pad_left"]
    hs = meta["scaled_h"]
    ws = meta["scaled_w"]
    mask[:, top:top + hs, left:left + ws] = 1.0
    return mask


def update_intrinsics_for_resize_pad(k, scale, pad_left, pad_top):
    if not torch.is_tensor(k):
        k = torch.tensor(k)

    k = k.clone().float()
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"camera_intrinsics must end with (3, 3), got {tuple(k.shape)}")

    k[..., 0, 0] *= float(scale)
    k[..., 1, 1] *= float(scale)
    k[..., 0, 2] = k[..., 0, 2] * float(scale) + float(pad_left)
    k[..., 1, 2] = k[..., 1, 2] * float(scale) + float(pad_top)
    return k


def validate_resize_pad_update(k_before, k_after, meta, valid_mask, atol=1e-6):
    fx0 = float(k_before[0, 0].item())
    fy0 = float(k_before[1, 1].item())
    cx0 = float(k_before[0, 2].item())
    cy0 = float(k_before[1, 2].item())

    fx1 = float(k_after[0, 0].item())
    fy1 = float(k_after[1, 1].item())
    cx1 = float(k_after[0, 2].item())
    cy1 = float(k_after[1, 2].item())

    scale = float(meta["scale"])
    pad_left = float(meta["pad_left"])
    pad_top = float(meta["pad_top"])

    if abs(fx1 - scale * fx0) > atol:
        raise AssertionError(f"fx mismatch: expected {scale * fx0}, got {fx1}")
    if abs(fy1 - scale * fy0) > atol:
        raise AssertionError(f"fy mismatch: expected {scale * fy0}, got {fy1}")
    if abs(cx1 - (scale * cx0 + pad_left)) > atol:
        raise AssertionError(f"cx mismatch: expected {scale * cx0 + pad_left}, got {cx1}")
    if abs(cy1 - (scale * cy0 + pad_top)) > atol:
        raise AssertionError(f"cy mismatch: expected {scale * cy0 + pad_top}, got {cy1}")

    u0 = cx0
    v0 = cy0
    u1 = scale * u0 + pad_left
    v1 = scale * v0 + pad_top

    x0 = (u0 - cx0) / fx0
    y0 = (v0 - cy0) / fy0
    x1 = (u1 - cx1) / fx1
    y1 = (v1 - cy1) / fy1

    if abs(x0 - x1) > atol or abs(y0 - y1) > atol:
        raise AssertionError(
            f"principal ray mismatch: old=({x0}, {y0}), new=({x1}, {y1})"
        )

    expected_area = int(meta["scaled_h"] * meta["scaled_w"])
    actual_area = int(valid_mask.sum().item())
    if expected_area != actual_area:
        raise AssertionError(
            f"valid_mask area mismatch: expected {expected_area}, got {actual_area}"
        )

    top = meta["pad_top"]
    left = meta["pad_left"]
    hs = meta["scaled_h"]
    ws = meta["scaled_w"]

    inner = valid_mask[:, top:top + hs, left:left + ws]
    if not torch.all(inner == 1):
        raise AssertionError("valid_mask inner valid region is not all ones")

    if top > 0 and not torch.all(valid_mask[:, :top, :] == 0):
        raise AssertionError("valid_mask top padding region is not zero")
    if left > 0 and not torch.all(valid_mask[:, :, :left] == 0):
        raise AssertionError("valid_mask left padding region is not zero")
    if top + hs < meta["new_h"] and not torch.all(valid_mask[:, top + hs:, :] == 0):
        raise AssertionError("valid_mask bottom padding region is not zero")
    if left + ws < meta["new_w"] and not torch.all(valid_mask[:, :, left + ws:] == 0):
        raise AssertionError("valid_mask right padding region is not zero")
    
class DatasetAdapter(torch.utils.data.Dataset):
    def apply_transform(transform, a, stack: bool = True):
        if isinstance(a, list):
            result = [
                DatasetAdapter.apply_transform(transform, i, stack) for i in a
            ]
            if stack:
                result = torch.stack(result)
            return result
        else:
            return transform(a)

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        transform_list: list,
        pop_list=None,
        enable_geometry_check: bool = False
    ):
        self.base_dataset = base_dataset
        self.transform_list = transform_list
        self.pop_list = pop_list
        self.enable_geometry_check = enable_geometry_check
        self._geometry_check_done = False

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        if isinstance(index, int):
            item = self.base_dataset[index]
            for i in self.transform_list:
                if i.get("is_dynamic_transform", False):
                    item = i["transform"](item)
                else:
                    item[i["new_key"]] = DatasetAdapter.apply_transform(
                        i["transform"], item[i["old_key"]],
                        i["stack"] if "stack" in i else True
                    )

            if self.pop_list is not None:
                for i in self.pop_list:
                    if i in item:
                        item.pop(i)

        elif isinstance(index, str):
            idx, num_frame, height, width = [
                int(val) for val in index.split("-")
            ]
            item = self.base_dataset[idx]

            start_f = random.randint(0, len(item["images"]) - num_frame)

            for k, v in item.items():
                if k != "fps" and k != "crossview_mask":
                    v = v[start_f:start_f + num_frame]
                item[k] = v

            image_resize_pad_meta = None
            image_valid_masks = None

            if "images" in item:
                src_images = item["images"]
                if not isinstance(src_images, list) or len(src_images) == 0:
                    raise ValueError("item['images'] must be a non-empty list")

                resized_padded_images = []
                image_resize_pad_meta = []
                image_valid_masks = []

                for img in src_images:
                    resized_padded_img, meta = resize_and_pad_image(
                        img, height, width, pad_value=0.0
                    )
                    valid_mask = make_valid_mask_from_meta(meta)
                    resized_padded_images.append(resized_padded_img)
                    image_resize_pad_meta.append(meta)
                    image_valid_masks.append(valid_mask)

                item["images"] = resized_padded_images

                valid_mask_tensor = torch.stack(image_valid_masks)
                image_size_before_resize_pad = torch.stack([
                    torch.tensor([m["orig_w"], m["orig_h"]], dtype=torch.long)
                    for m in image_resize_pad_meta
                ])
                image_size_after_resize_before_pad = torch.stack([
                    torch.tensor([m["scaled_w"], m["scaled_h"]], dtype=torch.long)
                    for m in image_resize_pad_meta
                ])
                image_size_tensor = torch.stack([
                    torch.tensor([m["new_w"], m["new_h"]], dtype=torch.long)
                    for m in image_resize_pad_meta
                ])

                if "camera_intrinsics" in item:
                    k_src_for_shape = item["camera_intrinsics"]
                    if not torch.is_tensor(k_src_for_shape):
                        k_src_for_shape = torch.tensor(k_src_for_shape)

                    if k_src_for_shape.ndim == 4:
                        t_count = k_src_for_shape.shape[0]
                        v_count = k_src_for_shape.shape[1]

                        if len(image_resize_pad_meta) != t_count * v_count:
                            raise ValueError(
                                f"image/meta count mismatch with camera_intrinsics: "
                                f"{len(image_resize_pad_meta)} vs {t_count * v_count}"
                            )

                        item["valid_mask"] = valid_mask_tensor.view(t_count, v_count, 1, height, width)
                        item["image_size_before_resize_pad"] = image_size_before_resize_pad.view(t_count, v_count, 2)
                        item["image_size_after_resize_before_pad"] = image_size_after_resize_before_pad.view(t_count, v_count, 2)
                        item["image_size"] = image_size_tensor.view(t_count, v_count, 2)

                    elif k_src_for_shape.ndim == 3:
                        t_count = k_src_for_shape.shape[0]

                        if len(image_resize_pad_meta) != t_count:
                            raise ValueError(
                                f"image/meta count mismatch with camera_intrinsics: "
                                f"{len(image_resize_pad_meta)} vs {t_count}"
                            )

                        item["valid_mask"] = valid_mask_tensor
                        item["image_size_before_resize_pad"] = image_size_before_resize_pad
                        item["image_size_after_resize_before_pad"] = image_size_after_resize_before_pad
                        item["image_size"] = image_size_tensor

                    else:
                        raise ValueError(
                            f"Unsupported camera_intrinsics shape for image_size/valid_mask reshape: "
                            f"{tuple(k_src_for_shape.shape)}"
                        )
                else:
                    item["valid_mask"] = valid_mask_tensor
                    item["image_size_before_resize_pad"] = image_size_before_resize_pad
                    item["image_size_after_resize_before_pad"] = image_size_after_resize_before_pad
                    item["image_size"] = image_size_tensor

                if "camera_intrinsics" in item:
                    k_src = item["camera_intrinsics"]
                    if not torch.is_tensor(k_src):
                        k_src = torch.tensor(k_src)

                    k_new = k_src.clone().float()

                    if k_new.ndim == 3:
                        if len(image_resize_pad_meta) != k_new.shape[0]:
                            raise ValueError(
                                f"camera_intrinsics frame count mismatch: "
                                f"{k_new.shape[0]} vs {len(image_resize_pad_meta)}"
                            )
                        for t in range(k_new.shape[0]):
                            meta = image_resize_pad_meta[t]
                            k_new[t] = update_intrinsics_for_resize_pad(
                                k_new[t], meta["scale"], meta["pad_left"], meta["pad_top"]
                            )

                    elif k_new.ndim == 4:
                        tv_count = k_new.shape[0] * k_new.shape[1]
                        if len(image_resize_pad_meta) != tv_count:
                            raise ValueError(
                                f"camera_intrinsics flattened frame-view count mismatch: "
                                f"{tv_count} vs {len(image_resize_pad_meta)}"
                            )
                        flat_idx = 0
                        for t in range(k_new.shape[0]):
                            for v in range(k_new.shape[1]):
                                meta = image_resize_pad_meta[flat_idx]
                                k_new[t, v] = update_intrinsics_for_resize_pad(
                                    k_new[t, v], meta["scale"], meta["pad_left"], meta["pad_top"]
                                )
                                flat_idx += 1
                    else:
                        raise ValueError(
                            f"Unsupported camera_intrinsics shape: {tuple(k_new.shape)}"
                        )

                    item["camera_intrinsics_before_resize_pad"] = k_src
                    item["camera_intrinsics"] = k_new

                    if self.enable_geometry_check and not self._geometry_check_done:
                        if k_src.ndim == 3:
                            validate_resize_pad_update(
                                k_src[0],
                                k_new[0],
                                image_resize_pad_meta[0],
                                item["valid_mask"][0]
                            )
                        else:
                            validate_resize_pad_update(
                                k_src[0, 0],
                                k_new[0, 0],
                                image_resize_pad_meta[0],
                                item["valid_mask"][0]
                            )

                        self._geometry_check_done = True
                        print(
                            "[DatasetAdapter] geometry+mask check passed: "
                            "resize+pad transform updates fx/fy/cx/cy and valid_mask correctly."
                        )

            for i in self.transform_list:
                old_key = i["old_key"]
                new_key = i["new_key"]
                stack = i.get("stack", True)

                if old_key == "images":
                    src = item[old_key]
                    if not isinstance(src, list):
                        raise ValueError(
                            f"{old_key} is expected to be a list in string-index path, got {type(src)}"
                        )

                    item[new_key] = torch.stack(src) if stack else src
                    continue

                if old_key in ["3dbox_images", "hdmap_images"]:
                    src = item[old_key]
                    if not isinstance(src, list):
                        raise ValueError(
                            f"{old_key} is expected to be a list in string-index path, got {type(src)}"
                        )

                    out = []
                    for img in src:
                        resized_padded_img, _ = resize_and_pad_image(
                            img, height, width, pad_value=0.0
                        )
                        out.append(resized_padded_img)

                    item[new_key] = torch.stack(out) if stack else out
                    continue

                if old_key == "proj_depth":
                    src = item["proj_depth"]
                    if isinstance(src, list):
                        item[new_key] = torch.stack([rs_bins_u16(x, height, width) for x in src])
                    else:
                        item[new_key] = rs_bins_u16(src, height, width)
                    continue

                if old_key == "proj_sem":
                    src = item["proj_sem"]
                    if isinstance(src, list):
                        item[new_key] = torch.stack([rs_bins_u16(x, height, width) for x in src])
                    else:
                        item[new_key] = rs_bins_u16(src, height, width)
                    continue

                if old_key == "proj_clr":
                    src = item["proj_clr"]
                    if isinstance(src, list):
                        out = [resize_clr_keep_invalid(c, height, width) for c in src]
                        item[new_key] = torch.stack(out) if stack else out
                    else:
                        item[new_key] = resize_clr_keep_invalid(src, height, width)
                    continue

                if getattr(i["transform"], "is_temporal_transform", False):
                    item[i["new_key"]] = DatasetAdapter.apply_temporal_transform(
                        i["transform"], item[i["old_key"]]
                    )
                else:
                    item[i["new_key"]] = DatasetAdapter.apply_transform(
                        i["transform"], item[i["old_key"]],
                        i["stack"] if "stack" in i else True
                    )

            if self.pop_list is not None:
                for i in self.pop_list:
                    if i in item:
                        item.pop(i)

        return item


class ConcatMotionDataset(torch.utils.data.Dataset):
    """Concatenate multiple datasets with given ratio. It is implemented for
    the training recipe in Vista(https://arxiv.org/abs/2405.17398).

    Args:
        datasets: a list of datasets.
        ratios: a list of ratios for each dataset.
    """

    def __init__(self, datasets: list, ratios: list):
        self.datasets = datasets
        self.full_size = math.ceil(
            max([
                len(dataset) / ratio
                for dataset, ratio in zip(datasets, ratios)
            ]))
        self.ranges = torch.cumsum(
            torch.tensor([int(ratio * self.full_size) for ratio in ratios]),
            dim=0)

    def __len__(self):
        return self.full_size

    def __getitem__(self, index):
        for i, range in enumerate(self.ranges):
            if index < range:
                return self.datasets[i][index % len(self.datasets[i])]

        raise Exception(f"invalid index {index}")


class CollateFnIgnoring():
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, item_list: list):
        ignored = [
            (key, [item.pop(key) for item in item_list])
            for key in self.keys
        ]
        result = torch.utils.data.default_collate(item_list)
        for key, value in ignored:
            result[key] = value

        return result


def find_nearest(list: list, value, return_item=False):
    i = bisect.bisect_left(list, value)
    if i == 0:
        pass
    elif i >= len(list):
        i = len(list) - 1
    else:
        diff_0 = value - list[i - 1]
        diff_1 = list[i] - value
        if i > 0 and diff_0 <= diff_1:
            i -= 1

    return list[i] if return_item else i



def find_nearest_2hz(timestamps, sdl, value, *, window=2, max_extra_us=0):
    i = bisect.bisect_left(timestamps, value)
    cand = [j for j in range(i - window, i + window + 1) if 0 <= j < len(timestamps)]

    best = min(
        cand,
        key=lambda j: (abs(timestamps[j] - value), 0 if timestamps[j] <= value else 1)
    )
    best_diff = abs(timestamps[best] - value)

    pref = [j for j in cand if len(sdl[j].get("token", "")) == 32]
    if pref:
        best_pref = min(
            pref,
            key=lambda j: (abs(timestamps[j] - value), 0 if timestamps[j] <= value else 1)
        )
        if abs(timestamps[best_pref] - value) <= best_diff + max_extra_us:
            return best_pref

    return best


def get_transform(rotation: list, translation: list, output_type: str = "np"):
    result = np.eye(4)
    result[:3, :3] = transforms3d.quaternions.quat2mat(rotation)
    result[:3, 3] = np.array(translation)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the get_transform()")


def make_intrinsic_matrix(fx_fy: list, cx_cy: list, output_type: str = "np"):
    result = np.diag(fx_fy + [1])
    result[:2, 2] = np.array(cx_cy)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the make_intrinsic_matrix()")


def project_line(
    a: np.array, b: np.array, near_z: float = 0.05, far_z: float = 512.0
):
    if (a[2] < near_z and b[2] < near_z) or (a[2] > far_z and b[2] > far_z):
        return None

    ca = a
    cb = b
    if a[2] >= near_z and b[2] < near_z:
        r = (near_z - b[2]) / (a[2] - b[2])
        cb = a * r + b * (1 - r)
    elif a[2] < near_z and b[2] >= near_z:
        r = (b[2] - near_z) / (b[2] - a[2])
        ca = a * r + b * (1 - r)

    if a[2] > far_z and b[2] <= far_z:
        r = (far_z - b[2]) / (a[2] - b[2])
        ca = a * r + b * (1 - r)
    elif a[2] <= far_z and b[2] > far_z:
        r = (b[2] - far_z) / (b[2] - a[2])
        cb = a * r + b * (1 - r)

    pa = ca[:2] / ca[2]
    pb = cb[:2] / cb[2]
    return (pa[0], pa[1], pb[0], pb[1])


def draw_edges_to_image(
    draw: ImageDraw.ImageDraw, points: np.array, edge_indices: list,
    pen_color: tuple, pen_width: int
):
    for a, b in edge_indices:
        xy = project_line(points[:, a], points[:, b])
        if xy is not None:
            draw.line(xy, fill=pen_color, width=pen_width)


def draw_3dbox_image(
    draw: ImageDraw.ImageDraw, view_transform: np.array,
    list_annotation_func, get_world_transform_func, get_annotation_label,
    pen_width: int, color_table: dict, corner_templates: list,
    edge_indices: list
):
    corner_templates_np = np.array(corner_templates).transpose()
    for sa in list_annotation_func():
        sa_label = get_annotation_label(sa)
        if sa_label in color_table:
            pen_color = tuple(color_table[sa_label])
            world_transform = get_world_transform_func(sa)
            p = view_transform @ world_transform @ corner_templates_np
            draw_edges_to_image(draw, p, edge_indices, pen_color, pen_width)


def align_image_description_crossview(caption_list: list, settings: dict):
    if "align_keys" in settings:
        for k in settings["align_keys"]:
            value_count = {}
            for i in caption_list:
                if i[k] not in value_count:
                    value_count[i[k]] = 0

                value_count[i[k]] += 1

            dominated_value = max(value_count, key=value_count.get)
            for i in caption_list:
                i[k] = dominated_value

    return caption_list


def make_image_description_string(
    caption_dict: dict, settings: dict, random_state: np.random.RandomState
):
    """Make the image description string from the caption dict with given
    settings.

    Args:
        caption_dict (dict): The caption dict contains textual descriptions of
            various categories such as time, environment, and more.
        settings (dict): The dict of settings to decide how to compose the
            final image descrption string.
            * selected_keys (list if exist): The value in the caption dict is
                used when its key in the list of selected keys.
            * reorder_keys (bool if exist): If set to True, the elements used
                to compose text descriptions in caption_dict will be shuffled.
            * drop_rates (dict if exist): The entries in the dict are the
                probabilities of the corresponding key elements in the
                caption_dict being dropped.
        random_state (np.random.RandomState): The random state for reproducible
            randomness.
    """
    default_image_description_keys = [
        "time", "weather", "environment", "objects", "image_description"
    ]
    selected_keys = settings.get(
        "selected_keys", default_image_description_keys)

    if "reorder_keys" in settings and settings["reorder_keys"]:
        new_order = random_state.permutation(len(selected_keys))
        selected_keys = [selected_keys[i] for i in new_order]

    if "drop_rates" in settings:
        drop = {
            k: random_state.rand() <= v
            for k, v in settings["drop_rates"].items()
        }
        selected_keys = [
            i for i in selected_keys
            if i not in drop or not drop[i]
        ]

    result = ". ".join([caption_dict[j] for j in selected_keys])
    # result = ". ".join([(caption_dict[j].split(",", 1)[0] if j == "weather" else caption_dict[j])
    #                     for j in selected_keys])
    return result


def add_stub_key_data(stub_key_data_dict, result: dict):
    """Add the stub key and data into the result dict.

    Args:
        stub_key_data_dict (dict or None): If set, the items are used to create
            stub item for the result dict. The value of this dict should be
            tuple. If the first item of the value tuple is "tensor", a tensor
            filled with the 3rd item in the shape of 2nd item is created as
            the stub data. Otherwise the 2nd item of the value tuple is
            deserialized as the stub data.
        result (dict): The result dict to insert created stub items.
    """

    if stub_key_data_dict is None:
        return

    for key, data in stub_key_data_dict.items():
        if key not in result.keys():
            if data[0] == "tensor":
                shape, value = data[1:]
                result[key] = value * torch.ones(shape)
            else:
                result[key] = data[1]



# -------------------------------- proj ----------------------------------





# ---------- png io ----------

def _safe_save_png(pil_img, p: str):
    tmp = p + ".tmp"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pil_img.save(tmp, format="PNG", compress_level=1, optimize=False)
    os.replace(tmp, p)

def _try_open_png(p: str):
    try:
        with Image.open(p) as im:
            im.load()
            return im.convert("RGB")
    except Exception:
        return None


# ---------- lock ----------

def _acquire_lock(lock: str, timeout=30, stale=120, sleep=0.02):
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale:
                    os.remove(lock)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout:
                raise TimeoutError(f"Lock timeout: {lock}")
            time.sleep(sleep)

def _release_lock(lock: str):
    try:
        os.remove(lock)
    except FileNotFoundError:
        pass


# ---------- u16 png io ----------

def _safe_save_u16_png(u16_img: np.ndarray, p: str):
    tmp = p + ".tmp"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.fromarray(u16_img.astype(np.uint16), mode="I;16").save(
        tmp, format="PNG", compress_level=1
    )
    os.replace(tmp, p)

def _try_open_u16_png(p: str):
    try:
        with Image.open(p) as im:
            im.load()
            if im.mode != "I;16":
                im = im.convert("I;16")
            return np.array(im, dtype=np.uint16)
    except Exception:
        return None


# ---------- cache subdir (for method wrapper) ----------

def ensure_cache_subdir(cache_root: str, subdir: str):
    root = os.path.abspath(cache_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"cache_root not found (won't create parent): {root}")

    sub = os.path.abspath(os.path.join(root, subdir))
    if os.path.dirname(sub) != root:
        raise ValueError(f"refuse to create nested cache dir: {sub}")

    os.makedirs(sub, exist_ok=True)
    return sub


# ---------- depth binning + vis ----------

def depth_to_logbins_u16(depth: np.ndarray, *, invalid=-300.0, n_bins=256, far_m=25.0, gamma=1.0) -> np.ndarray:
    assert 1 <= n_bins <= 65535
    out = np.zeros(depth.shape, np.uint16)

    m = depth != invalid
    if not np.any(m):
        return out

    far_m = float(far_m)
    near_m = far_m * 0.6
    near_frac = 0.85

    nb1 = max(1, int(round(n_bins * near_frac)))
    nb1 = min(nb1, n_bins - 1)
    nb2 = n_bins - nb1

    idx = np.flatnonzero(m)
    d = np.clip(depth[m].astype(np.float32), 0.0, far_m)

    m1 = d <= near_m
    if np.any(m1):
        x1 = np.log1p(d[m1]) / (np.log1p(near_m) + 1e-6)
        x1 = np.clip(x1, 0.0, 1.0)
        if gamma != 1.0:
            x1 = x1 ** float(gamma)
        out.reshape(-1)[idx[m1]] = (x1 * (nb1 - 1)).astype(np.uint16) + 1

    if nb2 > 0:
        m2 = ~m1
        if np.any(m2):
            dd = np.clip(d[m2] - near_m, 0.0, far_m - near_m)
            x2 = np.log1p(dd) / (np.log1p(far_m - near_m) + 1e-6)
            x2 = np.clip(x2, 0.0, 1.0)
            out.reshape(-1)[idx[m2]] = (x2 * (nb2 - 1)).astype(np.uint16) + 1 + nb1

    return out

def depth_to_linbins_u16(depth: np.ndarray, *, invalid=-300.0, n_bins=256, far_m=25.0) -> np.ndarray:
    assert 1 <= n_bins <= 65535
    out = np.zeros(depth.shape, np.uint16)

    m = depth != invalid
    if not np.any(m):
        return out

    d = np.clip(depth[m].astype(np.float32), 0.0, float(far_m))
    x = d / (float(far_m) + 1e-6)
    x = np.clip(x, 0.0, 1.0)

    out[m] = (x * (n_bins - 1)).astype(np.int32).astype(np.uint16) + 1
    return out

def visualize_bins_u16(bins_u16: np.ndarray, *, n_bins=256, invalid_bin=0, colormap=cv2.COLORMAP_TURBO):
    b = bins_u16.astype(np.int32)
    valid = b != int(invalid_bin)
    gray = np.zeros(b.shape, np.uint8)
    if np.any(valid):
        x = (b - 1) / max(1, (int(n_bins) - 1))
        x = np.clip(x, 0.0, 1.0)
        gray[valid] = (x[valid] * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(gray, colormap)
    vis[~valid] = (0, 0, 0)
    return vis


# ---------- downsample ----------

def downsample_depth_blockwise(depth_img, target_size, invalid=-300.0):
    m = (depth_img != invalid)
    d = cv2.resize(depth_img.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    m2 = cv2.resize(m.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.full(d.shape, invalid, np.float32)
    out[m2] = d[m2]
    return out

def downsample_clr_blockwise(img, target_size, blur_ksize=1):
    out = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    if blur_ksize and blur_ksize > 1:
        out = cv2.blur(out, (blur_ksize, blur_ksize))
    return out.astype(np.uint8)

# def downsample_clr_blockwise(img_u8, target_size, ks=5, beta=0.15):
#     """
#     img_u8: (H,W,3) uint8
#     target_size: (H2,W2)
#     ks: 全局平滑核大小(奇数)，越大越平滑
#     beta: 稀疏区稳定项，越大越不跳/越暗
#     """
#     H2, W2 = int(target_size[0]), int(target_size[1])

#     img = img_u8.astype(np.float32)
#     valid = (img_u8.sum(axis=2) > 0).astype(np.float32)  # (H,W)

#     num = cv2.resize(img * valid[..., None], (W2, H2), interpolation=cv2.INTER_AREA)
#     den = cv2.resize(valid, (W2, H2), interpolation=cv2.INTER_AREA)

#     if ks and ks > 1:
#         num = cv2.boxFilter(num, ddepth=-1, ksize=(ks, ks), normalize=True)
#         den = cv2.boxFilter(den, ddepth=-1, ksize=(ks, ks), normalize=True)
        
#     out = num / (den[..., None] + float(beta))

#     return np.clip(out, 0, 255).astype(np.uint8)
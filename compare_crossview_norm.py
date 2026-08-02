#!/usr/bin/env python3
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

CKPT_A = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/train_nuplantokenearly/"
    "checkpoints/24000.pth"
)
CKPT_B = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/train_nuplantokenearly/"
    "checkpoints/30000.pth"
)
OUTPUT_CSV = Path("crossview_norm_24000_vs_30000.csv")
TOP_K = 50

for checkpoint_path in (CKPT_A, CKPT_B):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

print(f"加载 A：{CKPT_A}")
checkpoint_a = torch.load(CKPT_A, map_location="cpu", weights_only=False)

print(f"加载 B：{CKPT_B}")
checkpoint_b = torch.load(CKPT_B, map_location="cpu", weights_only=False)

candidate_keys = (
    "state_dict",
    "model",
    "model_state_dict",
    "module",
    "network",
    "net",
)

state_dict_a = None
source_a = None
if isinstance(checkpoint_a, dict):
    direct_tensor_count = sum(
        isinstance(value, torch.Tensor)
        for value in checkpoint_a.values()
    )
    if direct_tensor_count > 0:
        state_dict_a = checkpoint_a
        source_a = "<checkpoint root>"
    else:
        for candidate_key in candidate_keys:
            candidate = checkpoint_a.get(candidate_key)
            if not isinstance(candidate, dict):
                continue
            tensor_count = sum(
                isinstance(value, torch.Tensor)
                for value in candidate.values()
            )
            if tensor_count > 0:
                state_dict_a = candidate
                source_a = candidate_key
                break

state_dict_b = None
source_b = None
if isinstance(checkpoint_b, dict):
    direct_tensor_count = sum(
        isinstance(value, torch.Tensor)
        for value in checkpoint_b.values()
    )
    if direct_tensor_count > 0:
        state_dict_b = checkpoint_b
        source_b = "<checkpoint root>"
    else:
        for candidate_key in candidate_keys:
            candidate = checkpoint_b.get(candidate_key)
            if not isinstance(candidate, dict):
                continue
            tensor_count = sum(
                isinstance(value, torch.Tensor)
                for value in candidate.values()
            )
            if tensor_count > 0:
                state_dict_b = candidate
                source_b = candidate_key
                break

if state_dict_a is None:
    top_keys = list(checkpoint_a.keys()) if isinstance(checkpoint_a, dict) else []
    raise KeyError(f"A 中未找到 state_dict，顶层键：{top_keys}")

if state_dict_b is None:
    top_keys = list(checkpoint_b.keys()) if isinstance(checkpoint_b, dict) else []
    raise KeyError(f"B 中未找到 state_dict，顶层键：{top_keys}")

normalized_a = {}
for original_key, value in state_dict_a.items():
    key = original_key
    while key.startswith("module.") or key.startswith("_orig_mod."):
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
    normalized_a[key] = value

normalized_b = {}
for original_key, value in state_dict_b.items():
    key = original_key
    while key.startswith("module.") or key.startswith("_orig_mod."):
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
    normalized_b[key] = value

keys_a = {
    key
    for key, value in normalized_a.items()
    if isinstance(value, torch.Tensor)
    and "crossview" in key.lower()
    and "norm" in key.lower()
}
keys_b = {
    key
    for key, value in normalized_b.items()
    if isinstance(value, torch.Tensor)
    and "crossview" in key.lower()
    and "norm" in key.lower()
}

common_keys = sorted(keys_a & keys_b)
only_a = sorted(keys_a - keys_b)
only_b = sorted(keys_b - keys_a)

print()
print(f"A state_dict 来源：{source_a}，总张量数：{len(normalized_a)}")
print(f"B state_dict 来源：{source_b}，总张量数：{len(normalized_b)}")
print(f"共同 crossview+norm 参数：{len(common_keys)}")
print(f"仅 A 存在：{len(only_a)}")
print(f"仅 B 存在：{len(only_b)}")

rows = []
shape_mismatch = []

for key in common_keys:
    tensor_a = normalized_a[key]
    tensor_b = normalized_b[key]

    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        shape_mismatch.append(
            (key, tuple(tensor_a.shape), tuple(tensor_b.shape))
        )
        continue

    flat_a = tensor_a.detach().float().cpu().reshape(-1)
    flat_b = tensor_b.detach().float().cpu().reshape(-1)
    diff = flat_b - flat_a
    abs_diff = diff.abs()

    l2_diff = diff.norm(p=2).item()
    reference_l2 = flat_a.norm(p=2).item()
    relative_l2 = l2_diff / max(reference_l2, 1e-12)

    cosine = float("nan")
    if flat_a.numel() > 1:
        norm_a = flat_a.norm(p=2).item()
        norm_b = flat_b.norm(p=2).item()
        if norm_a > 1e-12 and norm_b > 1e-12:
            cosine = F.cosine_similarity(
                flat_a.unsqueeze(0),
                flat_b.unsqueeze(0),
                dim=1,
            ).item()

    rows.append(
        {
            "key": key,
            "shape": str(tuple(tensor_a.shape)),
            "numel": flat_a.numel(),
            "a_mean": flat_a.mean().item(),
            "b_mean": flat_b.mean().item(),
            "a_std": flat_a.std(unbiased=False).item(),
            "b_std": flat_b.std(unbiased=False).item(),
            "l2_difference": l2_diff,
            "relative_l2_difference": relative_l2,
            "mean_absolute_difference": abs_diff.mean().item(),
            "max_absolute_difference": abs_diff.max().item(),
            "cosine_similarity": cosine,
            "changed_ratio_gt_1e-6": abs_diff.gt(1e-6).float().mean().item(),
        }
    )

rows.sort(
    key=lambda item: item["relative_l2_difference"],
    reverse=True,
)

if rows:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 120)
    print("按 relative L2 从大到小排列")
    print("=" * 120)
    print(
        f"{'relative L2':>13} "
        f"{'mean abs':>12} "
        f"{'max abs':>12} "
        f"{'cosine':>10}  parameter"
    )

    for row in rows[:TOP_K]:
        print(
            f"{row['relative_l2_difference']:13.6e} "
            f"{row['mean_absolute_difference']:12.6e} "
            f"{row['max_absolute_difference']:12.6e} "
            f"{row['cosine_similarity']:10.6f}  "
            f"{row['key']}"
        )

    all_diff_square = 0.0
    all_reference_square = 0.0
    all_numel = 0

    for key in common_keys:
        if tuple(normalized_a[key].shape) != tuple(normalized_b[key].shape):
            continue
        flat_a = normalized_a[key].detach().float().cpu().reshape(-1)
        flat_b = normalized_b[key].detach().float().cpu().reshape(-1)
        diff = flat_b - flat_a
        all_diff_square += diff.square().sum().item()
        all_reference_square += flat_a.square().sum().item()
        all_numel += flat_a.numel()

    global_l2 = all_diff_square ** 0.5
    global_relative_l2 = global_l2 / max(all_reference_square ** 0.5, 1e-12)
    global_rmse = (all_diff_square / max(all_numel, 1)) ** 0.5

    print()
    print(f"整体 L2 差异：{global_l2:.8e}")
    print(f"整体相对 L2：{global_relative_l2:.8e}")
    print(f"整体 RMSE：{global_rmse:.8e}")
    print(f"CSV 已保存：{OUTPUT_CSV.resolve()}")
else:
    print()
    print("没有找到可比较的 crossview+norm 参数。")
    print("下面列出参数名中含 crossview 的前 100 个键，检查实际命名：")
    crossview_candidates = sorted(
        key
        for key, value in normalized_a.items()
        if isinstance(value, torch.Tensor)
        and "crossview" in key.lower()
    )
    for key in crossview_candidates[:100]:
        print(key)

if only_a:
    print("\n仅 A 中存在：")
    for key in only_a:
        print(key)

if only_b:
    print("\n仅 B 中存在：")
    for key in only_b:
        print(key)

if shape_mismatch:
    print("\n形状不一致：")
    for key, shape_a, shape_b in shape_mismatch:
        print(f"{key}: A={shape_a}, B={shape_b}")

import os
import cv2
import torch
import numpy as np
if os.environ.get("ENABLE_DEBUGPY", "0") == "1":
    import debugpy

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        debugpy.listen(("0.0.0.0", 9876))
        print(
            "[debugpy] listening on 0.0.0.0:9876, waiting for VS Code to attach...",
            flush=True,
        )
        debugpy.wait_for_client()

import argparse
import json
import torch
import dwm.common


def customize_text(clip_text, preview_config):

    # text
    if preview_config["text"] is not None:
        text_config = preview_config["text"]

        if text_config["type"] == "add":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"] + k
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "replace":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"]
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "template":
            time = text_config["time"]
            weather = text_config["weather"]
            new_clip_text = \
                [
                    [
                        [
                            text_config["template"][time][weather][idx][0]
                            for idx, k in enumerate(j)
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        else:
            raise NotImplementedError(
                f"{text_config['type']}has not been implemented yet.")

        return new_clip_text

    else:

        return clip_text


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "-pc", "--preview-config-path", default=None, type=str,
        help="The config for preview setting")
    parser.add_argument(
        "-eic", "--export-item-config", default=False, type=bool,
        help="The flag to export the item config as JSON")
    return parser


def main():  # ========= 你要的 main 函数 + debug 在这里 =========
    # ========= 下面是你原来的全部代码，原封不动放进来 =========
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.preview_config_path is not None:
        with open(args.preview_config_path, "r", encoding="utf-8") as f:
            preview_config = json.load(f)
    else:
        preview_config = None

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp
    should_save = not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0

    # load the pipeline including the models
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device)
    if should_log:
        print("The pipeline is loaded.")

    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])

    preview_dataloader = torch.utils.data\
        .DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["preview_dataloader"])) if \
        "preview_dataloader" in config else None

    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    export_batch_except = ["vae_images"]
    output_path = args.output_path
    global_step = 0
    
    
    for i, batch in enumerate(preview_dataloader):
        ####调试#####################
        print("\n========== DEBUG BATCH ==========")
        print("keys:", batch.keys())
        ################################


        rank = int(os.environ.get("RANK", "0"))

        if rank == 0:
            debug_dir = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_layout_tokens"
            os.makedirs(debug_dir, exist_ok=True)

            B = 0

            boxes = batch["bbox_token_corners"].detach().cpu()      # [B,T,N,8,3]
            classes = batch["bbox_token_classes"].detach().cpu()    # [B,T,N]
            masks = batch["bbox_token_masks"].detach().cpu()        # [B,T,N]
            maps = batch["hdmap_bev_images"].detach().cpu()         # [B,T,3,H,W]
            imgs = batch["vae_images"].detach().cpu()               # [B,T,V,3,H,W]
            Ks = batch["camera_intrinsics"].detach().cpu()          # [B,T,V,3,3]
            cam_Ts = batch["camera_transforms"].detach().cpu()      # [B,T,V,4,4]

            stat_path = os.path.join(debug_dir, "layout_token_stats.txt")
            with open(stat_path, "w") as f:
                f.write("bbox_token_corners shape: {}\n".format(tuple(boxes.shape)))
                f.write("bbox_token_classes shape: {}\n".format(tuple(classes.shape)))
                f.write("bbox_token_masks shape: {}\n".format(tuple(masks.shape)))
                f.write("hdmap_bev_images shape: {}\n".format(tuple(maps.shape)))
                f.write("vae_images shape: {}\n".format(tuple(imgs.shape)))
                f.write("camera_intrinsics shape: {}\n".format(tuple(Ks.shape)))
                f.write("camera_transforms shape: {}\n".format(tuple(cam_Ts.shape)))
                f.write("\n")
                f.write("bbox valid total: {}\n".format(float(masks.sum())))
                f.write("bbox abs mean: {}\n".format(float(boxes.abs().mean())))
                f.write("map min: {}\n".format(float(maps.min())))
                f.write("map max: {}\n".format(float(maps.max())))
                f.write("map mean: {}\n".format(float(maps.mean())))
                f.write("map nonzero ratio: {}\n".format(float((maps.abs() > 1e-5).float().mean())))

            print("[layout debug] saved stats to:", stat_path)

            max_t = min(8, boxes.shape[1])
            max_v = min(8, imgs.shape[2])

            bev_scale = 6.4

            for t in range(max_t):
                bev_rgb = maps[B, t].permute(1, 2, 0).numpy()
                bev_rgb = np.clip(bev_rgb * 255.0, 0, 255).astype(np.uint8)

                map_only_bgr = bev_rgb[:, :, ::-1].copy()
                cv2.imwrite(
                    os.path.join(debug_dir, "map_bev_t{:02d}.png".format(t)),
                    map_only_bgr,
                )

                overlay = map_only_bgr.copy()
                bev_h, bev_w = overlay.shape[:2]
                center_x = bev_w * 0.5
                center_y = bev_h * 0.5

                valid_ids = torch.where(masks[B, t] > 0)[0]
                print("[layout debug] t={}, valid boxes={}".format(t, len(valid_ids)))

                for box_id in valid_ids.tolist():
                    corners = boxes[B, t, box_id].numpy()  # [8,3]

                    xy = corners[:, :2]
                    pts = np.zeros((xy.shape[0], 2), dtype=np.int32)
                    pts[:, 0] = np.round(xy[:, 0] * bev_scale + center_x).astype(np.int32)
                    pts[:, 1] = np.round(-xy[:, 1] * bev_scale + center_y).astype(np.int32)

                    inside = (
                        (pts[:, 0] >= 0) & (pts[:, 0] < bev_w) &
                        (pts[:, 1] >= 0) & (pts[:, 1] < bev_h)
                    )
                    if inside.sum() < 3:
                        continue

                    hull = cv2.convexHull(pts[inside].reshape(-1, 1, 2))
                    cv2.polylines(
                        overlay,
                        [hull],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    cls_id = int(classes[B, t, box_id].item())
                    label_xy = tuple(hull.reshape(-1, 2)[0].tolist())
                    cv2.putText(
                        overlay,
                        str(cls_id),
                        label_xy,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imwrite(
                    os.path.join(debug_dir, "map_plus_box_bev_t{:02d}.png".format(t)),
                    overlay,
                )

            for t in range(min(4, boxes.shape[1])):
                for v in range(max_v):
                    img_rgb = imgs[B, t, v].permute(1, 2, 0).numpy()
                    img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
                    canvas = img_rgb[:, :, ::-1].copy()

                    H, W = canvas.shape[:2]
                    K = Ks[B, t, v].numpy()
                    ego_from_cam = cam_Ts[B, t, v].numpy()
                    cam_from_ego = np.linalg.inv(ego_from_cam)

                    valid_ids = torch.where(masks[B, t] > 0)[0]

                    for box_id in valid_ids.tolist():
                        corners = boxes[B, t, box_id].numpy().astype(np.float32)

                        corners_h = np.concatenate(
                            [
                                corners,
                                np.ones((corners.shape[0], 1), dtype=np.float32),
                            ],
                            axis=1,
                        )

                        pts_cam = corners_h @ cam_from_ego.T
                        z = pts_cam[:, 2]

                        visible = z > 1e-3
                        if visible.sum() < 3:
                            continue

                        pts_cam_visible = pts_cam[visible, :3]
                        uvw = pts_cam_visible @ K.T
                        uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)

                        inside = (
                            (uv[:, 0] >= 0) & (uv[:, 0] < W) &
                            (uv[:, 1] >= 0) & (uv[:, 1] < H)
                        )

                        if inside.sum() < 3:
                            continue

                        uv_int = np.round(uv[inside]).astype(np.int32)
                        hull = cv2.convexHull(uv_int.reshape(-1, 1, 2))

                        cv2.polylines(
                            canvas,
                            [hull],
                            isClosed=True,
                            color=(0, 255, 0),
                            thickness=2,
                        )

                        cls_id = int(classes[B, t, box_id].item())
                        label_xy = tuple(hull.reshape(-1, 2)[0].tolist())
                        cv2.putText(
                            canvas,
                            str(cls_id),
                            label_xy,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    cv2.imwrite(
                        os.path.join(
                            debug_dir,
                            "bbox_proj_t{:02d}_v{:02d}.png".format(t, v),
                        ),
                        canvas,
                    )

            print("[layout debug] saved visualization to:", debug_dir)
        ################################
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: type={type(v)}")
        print("image_size sample:", batch["image_size"][0, 0, 0])
        #print("K_before:\n", batch["camera_intrinsics_before_resize_crop"][0, 0, 0])
        print("K_after:\n", batch["camera_intrinsics"][0, 0, 0])        
        #############检查crossview#############
        if "crossview_mask" in batch:
            print("\n--- crossview_mask sample ---")
            print(batch["crossview_mask"][0].int())  # 打印第一个
        #############检查相机数###########
        if "vae_images" in batch:
            print("\n--- camera check ---")
            print("vae_images shape:", batch["vae_images"].shape)
        ############检查cliptext#######
        if "clip_text" in batch:
            print("\n--- clip_text check ---")
            print(type(batch["clip_text"]))
            print("example:", batch["clip_text"][0][0])
        ################################
        if ddp:
            torch.distributed.barrier()

        if preview_config is not None:
            new_clip_text = customize_text(batch["clip_text"], preview_config)
            batch["clip_text"] = new_clip_text

        pipeline.preview_pipeline(
            batch, output_path, global_step)

        if args.export_item_config:
            with open(
                os.path.join(
                    output_path, "preview",
                    "{}.json".format(global_step)),
                "w", encoding="utf-8"
            ) as f:
                json.dump({
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                    if k not in export_batch_except
                }, f, indent=4)

        global_step += 1
        if should_log:
            print(f"preview: {global_step}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()  # 统一入口
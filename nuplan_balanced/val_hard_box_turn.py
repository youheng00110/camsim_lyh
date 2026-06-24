import os
import json
import pickle
import random
from collections import defaultdict

import numpy as np


PKL_PATH = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_prepo/mini_infos_val.pkl"
SAVE_DIR = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_balanced"

OUT_JSON = os.path.join(SAVE_DIR, "val_hard_box_turn_windows_metadata.json")
OUT_ALL_JSON = os.path.join(SAVE_DIR, "val_hard_box_turn_all_scored_windows_metadata.json")
OUT_SUMMARY = os.path.join(SAVE_DIR, "val_hard_box_turn_summary.json")

SEQUENCE_LENGTH = 20
FPS_STRIDE_TUPLES = [
    [6, 2, 0.5],
]

TOP_K = 480
MIN_DIST = 1.0

BOX_WEIGHT = 0.65
TURN_WEIGHT = 0.35

MAX_PER_SCENE = 4
MIN_START_GAP = 10

MAX_TIME_ERROR_RATIO = 0.5
RANDOM_SEED = 1234

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

with open(PKL_PATH, "rb") as f:
    obj = pickle.load(f)

infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj

scene_groups = defaultdict(list)
for info in infos:
    scene = info.get("db_name")
    timestamp = info.get("timestamp")
    if scene is None or timestamp is None:
        continue
    scene_groups[str(scene)].append(info)

for scene in scene_groups:
    scene_groups[scene].sort(key=lambda x: x["timestamp"])

windows = []

for scene, frames in scene_groups.items():
    timestamps = np.asarray([x["timestamp"] for x in frames], dtype=np.int64)
    frame_count = len(frames)

    if frame_count < SEQUENCE_LENGTH:
        continue

    for fps_stride_cfg in FPS_STRIDE_TUPLES:
        fps = float(fps_stride_cfg[0])
        stride = float(fps_stride_cfg[1])

        if fps == 0.0:
            step = max(1, int(stride))
            for start_idx in range(0, frame_count - SEQUENCE_LENGTH + 1, step):
                idxs = list(range(start_idx, start_idx + SEQUENCE_LENGTH))

                seq = [frames[i] for i in idxs]

                ego0 = np.asarray(seq[0]["ego2global"], dtype=np.float32)
                ego1 = np.asarray(seq[-1]["ego2global"], dtype=np.float32)

                dx = float(ego1[0, 3] - ego0[0, 3])
                dy = float(ego1[1, 3] - ego0[1, 3])
                dist = float(np.sqrt(dx * dx + dy * dy))

                if dist < MIN_DIST:
                    continue

                yaws = []
                vehicle_counts = []
                all_box_counts = []

                for frame in seq:
                    ego = np.asarray(frame["ego2global"], dtype=np.float32)
                    yaw = float(np.arctan2(ego[1, 0], ego[0, 0]))
                    yaws.append(yaw)

                    boxes = frame.get("gt_boxes", [])
                    names = frame.get("gt_names", [])
                    names = list(names) if isinstance(names, (list, tuple, np.ndarray)) else []

                    all_box_counts.append(len(boxes))

                    vehicle_count = 0
                    for name in names:
                        name = str(name).lower()
                        is_vehicle = (
                            "vehicle" in name
                            or "car" in name
                            or "truck" in name
                            or "bus" in name
                        )
                        if is_vehicle:
                            vehicle_count += 1

                    if len(names) == 0:
                        vehicle_count = len(boxes)

                    vehicle_counts.append(vehicle_count)

                yaws = np.unwrap(np.asarray(yaws, dtype=np.float32))
                turn_abs_deg = float(np.degrees(np.sum(np.abs(np.diff(yaws)))))
                turn_net_deg = float(np.degrees(abs(yaws[-1] - yaws[0])))

                windows.append({
                    "seq_id": scene,
                    "start_idx": int(idxs[0]),
                    "end_idx": int(idxs[-1]),
                    "start_timestamp": int(seq[0]["timestamp"]),
                    "end_timestamp": int(seq[-1]["timestamp"]),
                    "start_token": str(seq[0].get("lidarpc_token", "")),
                    "end_token": str(seq[-1].get("lidarpc_token", "")),
                    "fps": float(fps),
                    "stride": float(stride),
                    "angle": float(np.arctan2(dx, dy)),
                    "dist": dist,
                    "vehicle_box_mean": float(np.mean(vehicle_counts)),
                    "vehicle_box_max": int(np.max(vehicle_counts)),
                    "all_box_mean": float(np.mean(all_box_counts)),
                    "all_box_max": int(np.max(all_box_counts)),
                    "turn_abs_deg": turn_abs_deg,
                    "turn_net_deg": turn_net_deg,
                })

            continue

        dt_us = int(round(1e6 / fps))
        seq_dur_us = int((SEQUENCE_LENGTH - 1) * dt_us)

        t_begin = int(timestamps[0])
        t_last_begin = int(timestamps[-1] - seq_dur_us)

        if t_last_begin < t_begin:
            continue

        stride_us = dt_us if stride <= 0 else int(round(stride * 1e6))
        max_err_us = int(MAX_TIME_ERROR_RATIO * dt_us)

        t = t_begin
        while t <= t_last_begin:
            wanted = np.asarray([t + i * dt_us for i in range(SEQUENCE_LENGTH)], dtype=np.int64)

            idxs = []
            for w in wanted:
                pos = int(np.searchsorted(timestamps, w))
                if pos <= 0:
                    nearest = 0
                elif pos >= len(timestamps):
                    nearest = len(timestamps) - 1
                else:
                    left_err = abs(int(timestamps[pos - 1]) - int(w))
                    right_err = abs(int(timestamps[pos]) - int(w))
                    nearest = pos - 1 if left_err <= right_err else pos
                idxs.append(nearest)

            if len(set(idxs)) != SEQUENCE_LENGTH:
                t += stride_us
                continue

            picked = timestamps[np.asarray(idxs, dtype=np.int64)]
            if int(np.max(np.abs(picked - wanted))) > max_err_us:
                t += stride_us
                continue

            seq = [frames[i] for i in idxs]

            ego0 = np.asarray(seq[0]["ego2global"], dtype=np.float32)
            ego1 = np.asarray(seq[-1]["ego2global"], dtype=np.float32)

            dx = float(ego1[0, 3] - ego0[0, 3])
            dy = float(ego1[1, 3] - ego0[1, 3])
            dist = float(np.sqrt(dx * dx + dy * dy))

            if dist < MIN_DIST:
                t += stride_us
                continue

            yaws = []
            vehicle_counts = []
            all_box_counts = []

            for frame in seq:
                ego = np.asarray(frame["ego2global"], dtype=np.float32)
                yaw = float(np.arctan2(ego[1, 0], ego[0, 0]))
                yaws.append(yaw)

                boxes = frame.get("gt_boxes", [])
                names = frame.get("gt_names", [])
                names = list(names) if isinstance(names, (list, tuple, np.ndarray)) else []

                all_box_counts.append(len(boxes))

                vehicle_count = 0
                for name in names:
                    name = str(name).lower()
                    is_vehicle = (
                        "vehicle" in name
                        or "car" in name
                        or "truck" in name
                        or "bus" in name
                    )
                    if is_vehicle:
                        vehicle_count += 1

                if len(names) == 0:
                    vehicle_count = len(boxes)

                vehicle_counts.append(vehicle_count)

            yaws = np.unwrap(np.asarray(yaws, dtype=np.float32))
            turn_abs_deg = float(np.degrees(np.sum(np.abs(np.diff(yaws)))))
            turn_net_deg = float(np.degrees(abs(yaws[-1] - yaws[0])))

            windows.append({
                "seq_id": scene,
                "start_idx": int(idxs[0]),
                "end_idx": int(idxs[-1]),
                "start_timestamp": int(seq[0]["timestamp"]),
                "end_timestamp": int(seq[-1]["timestamp"]),
                "start_token": str(seq[0].get("lidarpc_token", "")),
                "end_token": str(seq[-1].get("lidarpc_token", "")),
                "fps": float(fps),
                "stride": float(stride),
                "angle": float(np.arctan2(dx, dy)),
                "dist": dist,
                "vehicle_box_mean": float(np.mean(vehicle_counts)),
                "vehicle_box_max": int(np.max(vehicle_counts)),
                "all_box_mean": float(np.mean(all_box_counts)),
                "all_box_max": int(np.max(all_box_counts)),
                "turn_abs_deg": turn_abs_deg,
                "turn_net_deg": turn_net_deg,
            })

            t += stride_us

if len(windows) == 0:
    raise RuntimeError("No valid val windows were found. Check PKL_PATH / FPS_STRIDE_TUPLES / SEQUENCE_LENGTH.")

vehicle_values = np.asarray([x["vehicle_box_mean"] for x in windows], dtype=np.float32)
turn_values = np.asarray([x["turn_abs_deg"] for x in windows], dtype=np.float32)

vehicle_order = np.argsort(np.argsort(vehicle_values)).astype(np.float32)
turn_order = np.argsort(np.argsort(turn_values)).astype(np.float32)

vehicle_rank = vehicle_order / max(1.0, float(len(vehicle_order) - 1))
turn_rank = turn_order / max(1.0, float(len(turn_order) - 1))

for i, window in enumerate(windows):
    window["vehicle_rank"] = float(vehicle_rank[i])
    window["turn_rank"] = float(turn_rank[i])
    window["hard_score"] = float(BOX_WEIGHT * vehicle_rank[i] + TURN_WEIGHT * turn_rank[i])

windows.sort(key=lambda x: x["hard_score"], reverse=True)

selected = []
scene_counter = defaultdict(int)
scene_starts = defaultdict(list)

for window in windows:
    scene = window["seq_id"]
    start_idx = int(window["start_idx"])

    if scene_counter[scene] >= MAX_PER_SCENE:
        continue

    too_close = False
    for old_start in scene_starts[scene]:
        if abs(start_idx - old_start) < MIN_START_GAP:
            too_close = True
            break

    if too_close:
        continue

    selected.append(window)
    scene_counter[scene] += 1
    scene_starts[scene].append(start_idx)

    if len(selected) >= TOP_K:
        break

if len(selected) < TOP_K:
    selected_keys = set()
    for window in selected:
        key = (window["seq_id"], window["start_idx"], window["end_idx"], window["fps"], window["stride"])
        selected_keys.add(key)

    for window in windows:
        key = (window["seq_id"], window["start_idx"], window["end_idx"], window["fps"], window["stride"])
        if key in selected_keys:
            continue

        selected.append(window)
        selected_keys.add(key)

        if len(selected) >= TOP_K:
            break

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

with open(OUT_ALL_JSON, "w", encoding="utf-8") as f:
    json.dump(windows, f, ensure_ascii=False, indent=2)

summary = {
    "pkl_path": PKL_PATH,
    "output": OUT_JSON,
    "all_scored_output": OUT_ALL_JSON,
    "total_candidates": len(windows),
    "selected": len(selected),
    "top_k": TOP_K,
    "sequence_length": SEQUENCE_LENGTH,
    "fps_stride_tuples": FPS_STRIDE_TUPLES,
    "min_dist": MIN_DIST,
    "box_weight": BOX_WEIGHT,
    "turn_weight": TURN_WEIGHT,
    "max_per_scene": MAX_PER_SCENE,
    "min_start_gap": MIN_START_GAP,
    "selected_vehicle_box_mean_avg": float(np.mean([x["vehicle_box_mean"] for x in selected])) if selected else 0.0,
    "selected_vehicle_box_max_avg": float(np.mean([x["vehicle_box_max"] for x in selected])) if selected else 0.0,
    "selected_all_box_mean_avg": float(np.mean([x["all_box_mean"] for x in selected])) if selected else 0.0,
    "selected_turn_abs_deg_avg": float(np.mean([x["turn_abs_deg"] for x in selected])) if selected else 0.0,
    "selected_turn_net_deg_avg": float(np.mean([x["turn_net_deg"] for x in selected])) if selected else 0.0,
    "selected_dist_avg": float(np.mean([x["dist"] for x in selected])) if selected else 0.0,
}

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))

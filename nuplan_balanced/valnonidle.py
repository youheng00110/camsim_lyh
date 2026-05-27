import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sys

sys.path.insert(
    0,
    "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src",
)

from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
)

from dwm.datasets.nuplan_splits import mini_val


DATA_ROOT = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_link/plan_data/mini"
MAP_ROOT = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_link/maps"
MAP_VERSION = "nuplan-maps-v1.0"

SAVE_DIR = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_balanced"

WINDOW_SIZE = 40
STEP_SIZE = 10
MIN_DIST = 1.0

os.makedirs(SAVE_DIR, exist_ok=True)

db_files = [f for f in os.listdir(DATA_ROOT) if f.endswith(".db")]
logs = [f.replace(".db", "") for f in db_files]
logs = [x for x in logs if x in set(mini_val)]

db_paths = [os.path.join(DATA_ROOT, f"{x}.db") for x in logs]

print("val db files:", len(db_paths))

db_wrapper = NuPlanDBWrapper(
    DATA_ROOT,
    MAP_ROOT,
    db_paths,
    MAP_VERSION,
)

all_frames = []

for log_name in tqdm(logs, desc="Extracting ego metadata"):
    db = db_wrapper.get_log_db(log_name)

    for i in range(0, len(db.lidar_pc), 2):
        lidar_pc = db.lidar_pc[i]

        ego_state = get_ego_state_for_lidarpc_token_from_db(
            db.load_path,
            lidar_pc.token,
        )

        pos = ego_state.center.array

        all_frames.append({
            "seq_id": log_name,
            "timestamp_micros": int(ego_state.time_us),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "lidar_token": lidar_pc.token,
        })

seq_groups = defaultdict(list)

for frame in all_frames:
    seq_groups[frame["seq_id"]].append(frame)

windows = []

for seq_id, frames in tqdm(seq_groups.items(), desc="Generating non-idle windows"):
    frames.sort(key=lambda x: x["timestamp_micros"])

    for i in range(0, len(frames) - WINDOW_SIZE, STEP_SIZE):
        start_f = frames[i]
        end_f = frames[i + WINDOW_SIZE - 1]

        dx = end_f["x"] - start_f["x"]
        dy = end_f["y"] - start_f["y"]

        dist = float(np.sqrt(dx ** 2 + dy ** 2))

        if dist < MIN_DIST:
            continue

        angle = float(np.arctan2(dx, dy))

        windows.append({
            "seq_id": seq_id,
            "start_idx": int(i),
            "end_idx": int(i + WINDOW_SIZE),
            "start_timestamp": int(start_f["timestamp_micros"]),
            "end_timestamp": int(end_f["timestamp_micros"]),
            "start_token": start_f["lidar_token"],
            "end_token": end_f["lidar_token"],
            "angle": angle,
            "dist": dist,
        })

out_path = os.path.join(SAVE_DIR, "val_nonidle_windows_metadata.json")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(windows, f, ensure_ascii=False, indent=2)

print("saved:", out_path)
print("non-idle windows:", len(windows))
print("first:", windows[0] if windows else None)
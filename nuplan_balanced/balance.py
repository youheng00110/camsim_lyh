import os
import json
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append("/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/src")

from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
)

from dwm.datasets.nuplan_splits import mini_train, mini_val

# ==============================
# 1. 配置
# ==============================

DATA_ROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuplan_link/plan_data/mini"
MAP_ROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuplan_link/maps"
MAP_VERSION = "nuplan-maps-v1.0"

SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuplan_balanced"

WINDOW_SIZE = 40
STEP_SIZE = 10
NUM_BINS = 36
SAMPLES_PER_BIN = 150

os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# 2. 加载 db
# ==============================

print("Loading nuPlan DB...")

db_files = [
    f for f in os.listdir(DATA_ROOT)
    if f.endswith(".db")
]

logs = [f.replace(".db", "") for f in db_files]

db_paths = [
    os.path.join(DATA_ROOT, f)
    for f in db_files
]

print("Total DB files:", len(db_paths))

db_wrapper = NuPlanDBWrapper(
    DATA_ROOT,
    MAP_ROOT,
    db_paths,
    MAP_VERSION
)

# ==============================
# 3. 读取 ego metadata
# ==============================

print("Extracting ego metadata...")

all_frames = []

for log_name in tqdm(logs):

    db = db_wrapper.get_log_db(log_name)

    for i in range(0, len(db.lidar_pc), 2): # 每 2 帧取 1 帧
        lidar_pc = db.lidar_pc[i]

        ego_state = get_ego_state_for_lidarpc_token_from_db(
            db.load_path,
            lidar_pc.token
        )

        pos = ego_state.center.array

        all_frames.append({

            "seq_id": log_name,

            "timestamp_micros": int(ego_state.time_us),

            "x": float(pos[0]),
            "y": float(pos[1]),

            "lidar_token": lidar_pc.token
        })


print("Total frames:", len(all_frames))

# ==============================
# 4. 按 sequence 分组
# ==============================

seq_groups = defaultdict(list)

for frame in all_frames:
    seq_groups[frame['seq_id']].append(frame)

# ==============================
# 5. 生成 windows
# ==============================

windows = []

print("Generating windows...")

for seq_id, frames in tqdm(seq_groups.items()):

    frames.sort(key=lambda x: x['timestamp_micros'])

    for i in range(0, len(frames) - WINDOW_SIZE, STEP_SIZE):

        start_f = frames[i]
        end_f = frames[i + WINDOW_SIZE - 1]

        dx = end_f['x'] - start_f['x']
        dy = end_f['y'] - start_f['y']

        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1.0:
            continue

        angle = np.arctan2(dx, dy)

        windows.append({

            "seq_id": seq_id,

            "start_idx": i,
            "end_idx": i + WINDOW_SIZE,

            "start_timestamp": start_f['timestamp_micros'],
            "end_timestamp": end_f['timestamp_micros'],

            "start_token": start_f["lidar_token"],
            "end_token": end_f["lidar_token"],

            "angle": float(angle),
            "dist": float(dist)
        })

print("Total windows:", len(windows))

# 保存全量
with open(os.path.join(SAVE_DIR, "all_windows_metadata.json"), "w") as f:
    json.dump(windows, f)

# ==============================
# 6. 角度分桶
# ==============================

bins = defaultdict(list)

bin_width = (2 * np.pi) / NUM_BINS

for w in tqdm(windows, desc="Binning windows"):

    theta = (w['angle'] + np.pi) % (2 * np.pi)

    bin_idx = int(theta // bin_width) % NUM_BINS

    bins[bin_idx].append(w)

# ==============================
# 7. 均匀采样
# ==============================

balanced_windows = []

print("Sampling windows...")

for b_idx in tqdm(range(NUM_BINS)):

    curr_bin_windows = bins[b_idx]

    if len(curr_bin_windows) == 0:
        continue

    num_to_sample = min(len(curr_bin_windows), SAMPLES_PER_BIN)

    sampled = random.sample(curr_bin_windows, num_to_sample)

    balanced_windows.extend(sampled)

# ==============================
# 8. 保存结果
# ==============================

output_path = os.path.join(SAVE_DIR, "balanced_windows_metadata.json")

with open(output_path, "w") as f:
    json.dump(balanced_windows, f, indent=2)

print("Balanced windows:", len(balanced_windows))

# ==============================
# 9. 分布可视化
# ==============================

sampled_angles = [w['angle'] for w in balanced_windows]

plt.figure(figsize=(10,5))

plt.hist(
    sampled_angles,
    bins=NUM_BINS,
    range=(-np.pi, np.pi),
    rwidth=0.8
)

plt.title("Balanced Motion Angle Distribution")

plt.xlabel("Angle(rad)")
plt.ylabel("Window Count")

plt.savefig(os.path.join(SAVE_DIR, "balanced_distribution.png"))

# ==============================
# 10. 极坐标对比
# ==============================

print("Drawing polar comparison...")

all_angles = [w['angle'] for w in windows]
balanced_angles = [w['angle'] for w in balanced_windows]

bin_edges = np.linspace(-np.pi, np.pi, NUM_BINS + 1)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

width = (2 * np.pi) / NUM_BINS

all_counts, _ = np.histogram(all_angles, bins=bin_edges)
balanced_counts, _ = np.histogram(balanced_angles, bins=bin_edges)

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, polar=True)

ax.bar(
    bin_centers,
    all_counts,
    width=width,
    color="lightgray",
    edgecolor="gray",
    alpha=0.6,
    label="Before"
)

ax.bar(
    bin_centers,
    balanced_counts,
    width=width,
    color="red",
    alpha=0.7,
    label="After"
)

ax.set_ylim(0, max(balanced_counts)*1.5)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.set_title("Motion Angle Distribution")

ax.legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "polar_distribution.png"),
    dpi=300
)

print("Done.")
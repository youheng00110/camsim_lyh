import os
import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# ==============================
# 1️⃣ 路径配置
# ==============================

DATAROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_link"

VERSION = "interp_12Hz_trainval"

SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_train_balanced"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 48
STEP_SIZE = 24

NUM_BINS = 36
SAMPLES_PER_BIN = 150

# ==============================
# 2️⃣ Camera 配置
# ==============================

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

NUM_CAM = len(CAMERAS)

# ==============================
# 3️⃣ 加载 nuScenes
# ==============================

print("Loading nuScenes...")

nusc = NuScenes(
    version=VERSION,
    dataroot=DATAROOT,
    verbose=False
)

train_scene_names = set(create_splits_scenes()['train'])

train_scenes = [
    scene for scene in nusc.scene
    if scene['name'] in train_scene_names
]

print("Total scenes in dataset:", len(nusc.scene))
print("Train scenes used:", len(train_scenes))


# ==============================
# 4️⃣ 构建 segment_tokens
# ==============================

def build_segment_tokens(nusc, start_sample, window_size):

    tokens = []

    curr_sample = start_sample

    for i in range(window_size):

        frame_tokens = []

        for cam in CAMERAS:

            sd_token = curr_sample["data"].get(cam)

            if sd_token is None:
                frame_tokens.append(None)
            else:
                frame_tokens.append(sd_token)

        tokens.append(frame_tokens)

        if curr_sample["next"] == "":
            break

        curr_sample = nusc.get("sample", curr_sample["next"])

    return tokens


# ==============================
# 5️⃣ 生成 windows
# ==============================

windows = []

print("Generating windows and calculating motion angles...")

for scene in tqdm(train_scenes, desc="Processing train scenes"):

    samples = []
    token = scene['first_sample_token']

    # 不跨 scene
    while token:

        sample = nusc.get('sample', token)
        samples.append(sample)

        token = sample['next']

    if len(samples) < WINDOW_SIZE:
        continue

    samples = sorted(samples, key=lambda x: x['timestamp'])

    for i in range(0, len(samples) - WINDOW_SIZE, STEP_SIZE):

        start_sample = samples[i]
        end_sample = samples[i + WINDOW_SIZE - 1]

        # =====================
        # ego pose
        # =====================

        sd_start = nusc.get('sample_data', start_sample['data']['LIDAR_TOP'])
        ego_start = nusc.get('ego_pose', sd_start['ego_pose_token'])

        sd_end = nusc.get('sample_data', end_sample['data']['LIDAR_TOP'])
        ego_end = nusc.get('ego_pose', sd_end['ego_pose_token'])

        x0, y0, _ = ego_start['translation']
        x1, y1, _ = ego_end['translation']

        dx = x1 - x0
        dy = y1 - y0

        dist = np.sqrt(dx**2 + dy**2)

        # 过滤几乎静止
        if dist < 1.0:
            continue

        # =====================
        # 计算运动角
        # =====================

        angle = np.arctan2(dx, dy)

        # =====================
        # 构建 token 序列
        # =====================

        segment_tokens = build_segment_tokens(
            nusc,
            start_sample,
            WINDOW_SIZE
        )

        if len(segment_tokens) != WINDOW_SIZE:
            continue

        windows.append({

            "scene_name": scene['name'],

            "start_sample_token": start_sample['token'],

            "end_sample_token": end_sample['token'],

            "segment_tokens": segment_tokens,

            "angle": float(angle),

            "dist": float(dist)
        })

print(f"Total valid windows generated: {len(windows)}")

with open(os.path.join(SAVE_DIR, "all_windows_metadata.json"), "w") as f:
    json.dump(windows, f)

# ==============================
# 6️⃣ 分桶
# ==============================

bins = defaultdict(list)

bin_width = (2 * np.pi) / NUM_BINS

print("Binning windows...")

for w in tqdm(windows):

    theta = (w['angle'] + np.pi) % (2 * np.pi)

    bin_idx = int(theta // bin_width) % NUM_BINS

    bins[bin_idx].append(w)

# ==============================
# 7️⃣ 均衡采样
# ==============================

balanced_windows = []

print(f"Sampling {SAMPLES_PER_BIN} per bin...")

for b_idx in tqdm(range(NUM_BINS)):

    curr_bin_windows = bins[b_idx]

    count = len(curr_bin_windows)

    if count == 0:
        print(f"Warning: Bin {b_idx} empty")
        continue

    num_to_sample = min(count, SAMPLES_PER_BIN)

    sampled = random.sample(curr_bin_windows, num_to_sample)

    balanced_windows.extend(sampled)

    print(f"Bin {b_idx}: Found {count}, Sampled {num_to_sample}")

output_path = os.path.join(SAVE_DIR, "balanced_windows_metadata.json")

with open(output_path, "w") as f:
    json.dump(balanced_windows, f, indent=2)

print(f"\nFinal balanced dataset size: {len(balanced_windows)}")
print(f"Saved to: {output_path}")

# ==============================
# 8️⃣ 直方图验证
# ==============================

sampled_angles = [w['angle'] for w in balanced_windows]

plt.figure(figsize=(10,5))

plt.hist(
    sampled_angles,
    bins=NUM_BINS,
    range=(-np.pi, np.pi),
    rwidth=0.8
)

plt.title("nuScenes Balanced Motion Angle Distribution")

plt.xlabel("Angle (rad)")
plt.ylabel("Window Count")

plt.savefig(
    os.path.join(SAVE_DIR, "balanced_distribution_check.png")
)

# ==============================
# 9️⃣ 极坐标对比
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
    alpha=0.6,
    label="Before balancing"
)

ax.bar(
    bin_centers,
    balanced_counts,
    width=width,
    color="red",
    alpha=0.7,
    label="After balancing"
)

ax.set_ylim(0, 200)

ax.set_theta_zero_location("N")

ax.set_theta_direction(-1)

ax.set_title("nuScenes Motion Angle Distribution (Polar)")

ax.legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "polar_distribution_comparison.png"),
    dpi=300
)

print("Done.")
import os
import json
import math
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


# ===============================
# 路径配置
# ===============================
DATAROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_link"

SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_val_sort"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# 固定 window
# ===============================
WINDOW_SIZE = 20
STEP_SIZE = 10


# ===============================
# 相机顺序
# ===============================
CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT"
]


# ===============================
# 分类阈值
# ===============================

IDLE_DISP_THR = 6.0
IDLE_VMEAN_THR = 1.2
IDLE_CUMYAW_THR = 12.0

AGG_MAX_ACC = 2.5
AGG_MAX_DEC = -3.5
AGG_CUMYAW = 50.0
AGG_VMEAN = 5.0

TURN_CUMYAW = 30.0

STRAIGHT_CUMYAW = 15.0
STRAIGHT_LATERAL = 1.8


# ===============================
# 工具函数
# ===============================

def quat_to_yaw(q):
    w, x, y, z = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def angle_diff(a, b):
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def cumulative_abs_yaw(yaws):
    total = 0
    for i in range(len(yaws) - 1):
        total += abs(angle_diff(yaws[i + 1], yaws[i]))
    return math.degrees(total)


def compute_lateral_shift(poses, yaws):

    p0 = np.array(poses[0])
    p1 = np.array(poses[-1])
    theta0 = yaws[0]

    delta = p1 - p0

    lateral_vec = np.array([
        -np.sin(theta0),
        np.cos(theta0)
    ])

    lateral_shift = np.dot(delta, lateral_vec)

    return float(abs(lateral_shift))


# ===============================
# 构建 segment_tokens [T,C]
# ===============================

def build_segment_tokens(nusc, sample_tokens):

    segment_tokens = []

    for st in sample_tokens:

        sample = nusc.get("sample", st)

        cam_tokens = []

        for cam in CAMERA_ORDER:

            sd_token = sample["data"].get(cam, None)

            if sd_token is None:
                cam_tokens.append(None)
            else:
                cam_tokens.append(sd_token)

        segment_tokens.append(cam_tokens)

    return segment_tokens


# ===============================
# 加载 nuScenes
# ===============================

print("Loading nuScenes...")

nusc = NuScenes(
    version='v1.0-trainval',
    dataroot=DATAROOT,
    verbose=False
)

val_scene_names = set(create_splits_scenes()['val'])

val_scenes = [s for s in nusc.scene if s['name'] in val_scene_names]

print("Val scenes:", len(val_scenes))


classified_windows = []
reverse_index = defaultdict(list)


# ===============================
# 主循环
# ===============================

for scene in tqdm(val_scenes, desc="Processing val scenes"):

    samples = []

    token = scene['first_sample_token']

    while token:

        sample = nusc.get('sample', token)

        samples.append(sample)

        token = sample['next']

    samples = sorted(samples, key=lambda x: x['timestamp'])

    if len(samples) < WINDOW_SIZE:
        continue

    for i in range(0, len(samples) - WINDOW_SIZE + 1, STEP_SIZE):

        window_samples = samples[i:i + WINDOW_SIZE]

        poses = []
        yaws = []
        timestamps = []
        sample_tokens = []

        for s in window_samples:

            sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])

            ego = nusc.get('ego_pose', sd['ego_pose_token'])

            x, y, _ = ego['translation']

            yaw = quat_to_yaw(ego['rotation'])

            poses.append((x, y))
            yaws.append(yaw)
            timestamps.append(s['timestamp'])
            sample_tokens.append(s['token'])

        # ======================
        # 位移
        # ======================

        x0, y0 = poses[0]
        x1, y1 = poses[-1]

        dx = x1 - x0
        dy = y1 - y0

        dist = math.hypot(dx, dy)

        # ======================
        # lateral shift
        # ======================

        lateral_shift = compute_lateral_shift(poses, yaws)

        # ======================
        # 速度
        # ======================

        v_mag = []
        v_long = []

        start_yaw = yaws[0]

        for j in range(len(poses) - 1):

            dt = (timestamps[j + 1] - timestamps[j]) / 1e6

            if dt <= 0:
                continue

            xa, ya = poses[j]
            xb, yb = poses[j + 1]

            vx = (xb - xa) / dt
            vy = (yb - ya) / dt

            v_mag.append(math.hypot(vx, vy))

            v_long.append(
                vx * math.cos(start_yaw) +
                vy * math.sin(start_yaw)
            )

        if len(v_mag) == 0:
            continue

        v_mean = float(np.mean(v_mag))

        acc_long = []

        for j in range(len(v_long) - 1):

            dt = (timestamps[j + 1] - timestamps[j]) / 1e6

            if dt <= 0:
                continue

            acc_long.append((v_long[j + 1] - v_long[j]) / dt)

        max_acc = max(acc_long) if acc_long else 0
        max_dec = min(acc_long) if acc_long else 0

        cum_yaw = cumulative_abs_yaw(yaws)

        # ======================
        # 分类逻辑
        # ======================

        if (max_acc > AGG_MAX_ACC or
            max_dec < AGG_MAX_DEC or
            (cum_yaw > AGG_CUMYAW and v_mean > AGG_VMEAN)):

            label = "aggressive"

        elif (dist < IDLE_DISP_THR and
              v_mean < IDLE_VMEAN_THR and
              cum_yaw < IDLE_CUMYAW_THR):

            label = "idle"

        elif cum_yaw > TURN_CUMYAW:

            label = "turning"

        elif (cum_yaw < STRAIGHT_CUMYAW and
              lateral_shift < STRAIGHT_LATERAL):

            label = "straight"

        else:

            label = "turning"

        # ======================
        # 构建 segment_tokens
        # ======================

        segment_tokens = build_segment_tokens(nusc, sample_tokens)

        # ======================
        # JSON entry
        # ======================

        window_meta = {

            "scene_name": scene['name'],

            "start_sample_token": window_samples[0]['token'],
            "end_sample_token": window_samples[-1]['token'],

            "segment_tokens": segment_tokens,

            "angle": float(cum_yaw),
            "dist": float(dist),

            "timestamps": timestamps,
            "sample_tokens": sample_tokens,

            "metrics": {
                "distance_m": float(dist),
                "v_mean_m_s": float(v_mean),
                "max_acc_m_s2": float(max_acc),
                "max_dec_m_s2": float(max_dec),
                "cum_yaw_deg": float(cum_yaw),
                "lateral_shift_m": float(lateral_shift)
            },

            "label": label
        }

        classified_windows.append(window_meta)

        idx = len(classified_windows) - 1

        for st in sample_tokens:
            reverse_index[st].append(idx)


# ===============================
# 保存
# ===============================

print("Total windows generated:", len(classified_windows))

with open(os.path.join(SAVE_DIR, "classified_val_windows.json"), "w") as f:
    json.dump(classified_windows, f, indent=2)

with open(os.path.join(SAVE_DIR, "reverse_index.json"), "w") as f:
    json.dump(reverse_index, f, indent=2)

print("Saved to:", SAVE_DIR)

print(
    "Label distribution:",
    Counter([w["label"] for w in classified_windows])
)
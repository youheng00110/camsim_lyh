import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# ==============================
# 1️⃣ 路径配置
# ==============================
DATAROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_link"
SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh"
LOOK_AHEAD_STEPS = 10   # 2Hz × 5秒 ≈ 10帧

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading nuScenes...")
nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=False)
train_scene_names = set(create_splits_scenes()['train'])

relative_coords = []

print("Calculating ego-centric future trajectories...")

# ==============================
# 2️⃣ scene-wise 处理（关键）
# ==============================
for scene in nusc.scene:
    if scene['name'] not in train_scene_names:
        continue

    # 收集当前 scene 的所有 sample
    samples = []
    token = scene['first_sample_token']

    while token:
        sample = nusc.get('sample', token)
        samples.append(sample)
        token = sample['next']

    if len(samples) < LOOK_AHEAD_STEPS + 1:
        continue

    # 按时间排序（保险）
    samples = sorted(samples, key=lambda x: x['timestamp'])

    # ==========================
    # 3️⃣ 在 scene 内部滑动窗口
    # ==========================
    for i in range(len(samples) - LOOK_AHEAD_STEPS):
        curr = samples[i]

        sd = nusc.get('sample_data', curr['data']['LIDAR_TOP'])
        ego = nusc.get('ego_pose', sd['ego_pose_token'])

        curr_x, curr_y, _ = ego['translation']
        w, xq, yq, zq = ego['rotation']

        # yaw（East-based）
        yaw = np.arctan2(
            2 * (w * zq + xq * yq),
            1 - 2 * (yq**2 + zq**2)
        )

        # 旋转矩阵分量
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 未来轨迹
        for j in range(1, LOOK_AHEAD_STEPS + 1):
            fut = samples[i + j]

            sd_fut = nusc.get('sample_data', fut['data']['LIDAR_TOP'])
            ego_fut = nusc.get('ego_pose', sd_fut['ego_pose_token'])

            fut_x, fut_y, _ = ego_fut['translation']

            dx = fut_x - curr_x
            dy = fut_y - curr_y

            # ===== 坐标转换 =====
            # Forward = +Y
            # Right   = +X

            local_forward = dx * cos_yaw + dy * sin_yaw
            local_right   = dx * sin_yaw - dy * cos_yaw

            relative_coords.append([local_right, local_forward])

relative_coords = np.array(relative_coords)

print(f"Total trajectory points: {len(relative_coords)}")

# ==============================
# 4️⃣ 绘制 Heatmap
# ==============================
print("Plotting heatmap...")

plt.figure(figsize=(10, 12))

hb = plt.hexbin(
    relative_coords[:, 0],
    relative_coords[:, 1],
    gridsize=150,
    cmap='YlOrRd',
    bins='log',
    mincnt=1
)

cb = plt.colorbar(hb, label='log10(frame count)')

plt.axvline(0, linestyle='--', alpha=0.3)
plt.axhline(0, linestyle='--', alpha=0.3)

plt.xlabel('Lateral Distance (Left <--- 0 ---> Right) [m]')
plt.ylabel('Forward Distance [m]')
plt.title('nuScenes Ego-centric Future Trajectory Heatmap (5s, Train Split)')

plt.xlim(-30, 30)
plt.ylim(-5, 80)

plt.grid(True, alpha=0.15)
plt.gca().set_facecolor('white')

save_path = os.path.join(SAVE_DIR, "nuscenes_turning_heatmap_red.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Success! Saved to {save_path}")
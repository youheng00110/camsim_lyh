import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# ===============================
# 1️⃣ 路径配置
# ===============================
DATAROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_link"
SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# 2️⃣ 初始化
# ===============================
print("Loading nuScenes...")
nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=False)
train_scene_names = set(create_splits_scenes()['train'])

all_heading = []
all_speed = []
all_curvature = []

# ===============================
# 3️⃣ scene-wise 计算
# ===============================
print("Processing train scenes...")

for scene in nusc.scene:
    if scene['name'] not in train_scene_names:
        continue

    xs, ys, yaws, ts = [], [], [], []
    token = scene['first_sample_token']

    while token:
        sample = nusc.get('sample', token)
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego = nusc.get('ego_pose', sd['ego_pose_token'])

        x, y, _ = ego['translation']
        w, xq, yq, zq = ego['rotation']

        # quaternion → yaw (global frame, East-based)
        yaw = np.arctan2(
            2 * (w * zq + xq * yq),
            1 - 2 * (yq**2 + zq**2)
        )

        xs.append(x)
        ys.append(y)
        yaws.append(yaw)
        ts.append(sample['timestamp'])

        token = sample['next']

    if len(xs) < 2:
        continue

    xs = np.array(xs)
    ys = np.array(ys)
    yaws = np.array(yaws)
    ts = np.array(ts)

    # -------- 速度计算 --------
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx**2 + dy**2)

    dt = np.diff(ts) / 1e6  # microsecond → second
    dt = np.maximum(dt, 1e-3)

    speed = ds / dt

    # -------- 曲率计算 --------
    d_yaw = np.diff(yaws)

    # 角度 wrap 到 [-pi, pi]
    d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi

    curvature = d_yaw / np.maximum(ds, 1e-3)

    all_heading.append(yaws)
    all_speed.append(speed)
    all_curvature.append(curvature)

# ===============================
# 4️⃣ 合并
# ===============================
heading = np.concatenate(all_heading)
speed = np.concatenate(all_speed)
curvature = np.concatenate(all_curvature)

# East-based → North-based
heading = heading - np.pi / 2
heading = (heading + np.pi) % (2 * np.pi) - np.pi

print(f"Total heading samples: {len(heading)}")
print(f"Total speed samples: {len(speed)}")
print(f"Total curvature samples: {len(curvature)}")

# ===============================
# 5️⃣ 极坐标 Heading
# ===============================
print("Plotting Polar Heading...")

bins = 36
edges = np.linspace(-np.pi, np.pi, bins + 1)
counts, _ = np.histogram(heading, bins=edges)
angles = (edges[:-1] + edges[1:]) / 2

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

ax.set_theta_zero_location('N')   # 0° 在上
ax.set_theta_direction(-1)        # 顺时针

ax.bar(angles, counts, width=2*np.pi/bins)
ax.set_title("nuScenes Train Heading (0°=North)")

plt.savefig(os.path.join(SAVE_DIR, "nuscenes_heading_polar.png"), dpi=300)
plt.close()

# ===============================
# 6️⃣ Heading 直方图
# ===============================
plt.figure(figsize=(8, 6))
plt.hist(heading, bins=100)
plt.xlabel("Heading (rad)")
plt.ylabel("Count")
plt.title("Heading Distribution")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "nuscenes_heading_hist.png"), dpi=300)
plt.close()

# ===============================
# 7️⃣ Speed 分布
# ===============================
plt.figure(figsize=(8, 6))
plt.hist(speed, bins=100)
plt.xlabel("Speed (m/s)")
plt.ylabel("Count")
plt.title("Speed Distribution")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "nuscenes_speed.png"), dpi=300)
plt.close()

# ===============================
# 8️⃣ Curvature 分布
# ===============================
plt.figure(figsize=(8, 6))
plt.hist(curvature, bins=100, range=(-0.2, 0.2))
plt.xlabel("Curvature (1/m)")
plt.ylabel("Count")
plt.title("Curvature Distribution (Clipped)")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "nuscenes_curvature.png"), dpi=300)
plt.close()

print(f"\nAll plots saved to {SAVE_DIR}")
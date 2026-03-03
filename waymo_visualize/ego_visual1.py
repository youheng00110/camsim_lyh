import numpy as np
import matplotlib
# 强制使用 Agg 后端，防止在无显示器的服务器上报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# 1. 路径配置
METADATA_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/prepro_waymo/waymo_ego_metadata.json"
SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh"

# 2. 加载元数据
print("Loading metadata...")
with open(METADATA_PATH, 'r') as f:
    all_ego = json.load(f)

# 提取变量并转为 numpy 数组加速计算
x = np.array([e['x'] for e in all_ego])
y = np.array([e['y'] for e in all_ego])
heading = np.array([e['heading'] for e in all_ego])
curvature = np.array([e['curvature'] for e in all_ego])
speed = np.array([e['speed'] for e in all_ego])

print(f"Total frames: {len(x)}")

# --- 绘图 1: 行车轨迹分布 (X-Y) ---
print("Plotting Trajectory...")
plt.figure(figsize=(10, 8))
# 使用 hexbin 代替 scatter，能更直观看到哪里数据最密集
# gridsize 越大越精细
hb = plt.hexbin(x - np.mean(x), y - np.mean(y), gridsize=100, cmap='inferno', bins='log')
plt.colorbar(hb, label='log10(sample count)')
plt.axis('equal') # 保证 X Y 轴比例一致，不扭曲轨迹
plt.xlabel('Relative X (m)')
plt.ylabel('Relative Y (m)')
plt.title('Trajectory Point Density (Zero-centered)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "waymo_trajectory.png"), dpi=300)
plt.close()

# --- 绘图 2: 航向角分布 ---
print("Plotting Heading...")
plt.figure(figsize=(8, 6))
plt.hist(heading, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Heading Angle (rad)')
plt.ylabel('Count')
plt.title('Heading Angle Distribution (-pi to pi)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "waymo_heading_dist.png"), dpi=300)
plt.close()

# --- 绘图 3: 曲率分布 (带范围限制) ---
print("Plotting Curvature...")
plt.figure(figsize=(8, 6))
# 过滤掉异常大的曲率值（只看 -0.2 到 0.2 之间，这是正常的转弯范围）
clean_curvature = curvature[~np.isnan(curvature)]
plt.hist(clean_curvature, bins=100, range=(-0.2, 0.2), color='salmon', edgecolor='black', alpha=0.7)
plt.xlabel('Curvature (1/m)')
plt.ylabel('Count')
plt.title('Curvature Distribution (Clipped to [-0.2, 0.2])')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "waymo_curvature_dist.png"), dpi=300)
plt.close()

# --- 绘图 4: 速度分布 (补充项，很有参考价值) ---
print("Plotting Speed...")
plt.figure(figsize=(8, 6))
plt.hist(speed, bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
plt.xlabel('Speed (m/s)')
plt.ylabel('Count')
plt.title('Speed Distribution')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "waymo_speed_dist.png"), dpi=300)
plt.close()

print(f"All plots saved to {SAVE_DIR}")
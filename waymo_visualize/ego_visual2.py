import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

# 1. 配置路径
METADATA_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/prepro_waymo/waymo_ego_metadata.json"
SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh"
LOOK_AHEAD_STEPS = 50  # 预测未来 5 秒 (10Hz)

print("Loading metadata...")
with open(METADATA_PATH, 'r') as f:
    data = json.load(f)

# 按序列分组
sequences = defaultdict(list)
for entry in data:
    sequences[entry['seq_id']].append(entry)

relative_coords = []

print("Calculating Trajectories (Forward = +Y, Right = +X)...")
for seq_id, frames in sequences.items():
    frames = sorted(frames, key=lambda x: x['timestamp_micros'])
    
    for i in range(len(frames) - LOOK_AHEAD_STEPS):
        curr = frames[i]
        curr_x, curr_y, curr_yaw = curr['x'], curr['y'], curr['heading']
        
        cos_yaw = np.cos(curr_yaw)
        sin_yaw = np.sin(curr_yaw)

        for j in range(1, LOOK_AHEAD_STEPS + 1):
            fut = frames[i + j]
            dx = fut['x'] - curr_x
            dy = fut['y'] - curr_y
            
            # --- 修正后的坐标转换公式 ---
            # 使自车当前航向对应 Y 轴正方向 (Forward)
            # 使自车右侧对应 X 轴正方向 (Right)
            local_forward = dx * cos_yaw + dy * sin_yaw
            local_right = dx * sin_yaw - dy * cos_yaw
            
            # 我们要画在图上的：横轴是 Right (X), 纵轴是 Forward (Y)
            relative_coords.append([local_right, local_forward])

relative_coords = np.array(relative_coords)

# 2. 绘图
print(f"Plotting Heatmap with {len(relative_coords)} points...")
plt.figure(figsize=(10, 12))

# cmap='YlOrRd' 代表 越密集越红 (Yellow -> Orange -> Red)
# bins='log' 处理密集数据的对比度
hb = plt.hexbin(
    relative_coords[:, 0], 
    relative_coords[:, 1], 
    gridsize=150, 
    cmap='YlOrRd', 
    bins='log', 
    mincnt=1
)

# 装饰图表
cb = plt.colorbar(hb, label='log10(frame count)')
plt.axvline(0, color='black', linestyle='--', alpha=0.3) # 纵向中心线（直行线）
plt.axhline(0, color='black', linestyle='--', alpha=0.3) # 起始点横线

plt.xlabel('Lateral Distance (Left <--- 0 ---> Right) [m]')
plt.ylabel('Forward Distance [m]')
plt.title('Ego-centric Trajectory Heatmap (Forward is UP)')

# 调整显示范围
# 通常 5 秒行驶距离在 0-80 米左右，侧向偏移在 -30 到 30 米左右
plt.xlim(-30, 30)
plt.ylim(-5, 80)

plt.grid(True, alpha=0.15)
plt.gca().set_facecolor('white') # 背景设为白色更清晰

save_path = os.path.join(SAVE_DIR, "waymo_turning_heatmap_red.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Success! Saved to {save_path}")
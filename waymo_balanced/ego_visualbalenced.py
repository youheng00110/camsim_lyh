import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

# 1. 路径配置
METADATA_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/prepro_waymo/waymo_ego_metadata.json"
BALANCED_WINDOWS_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/waymo_balanced/balanced_windows_metadata.json"
SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh"

print("Loading data...")
with open(METADATA_PATH, 'r') as f:
    all_frames = json.load(f)
with open(BALANCED_WINDOWS_PATH, 'r') as f:
    balanced_windows = json.load(f)

seq_to_frames = defaultdict(list)
for f in all_frames:
    seq_to_frames[f['seq_id']].append(f)
for seq_id in seq_to_frames:
    seq_to_frames[seq_id].sort(key=lambda x: x['timestamp_micros'])

relative_coords = []
print(f"Processing {len(balanced_windows)} windows (4.0s/40 frames)...")

for win in balanced_windows:
    seq_id, start_idx, end_idx = win['seq_id'], win['start_idx'], win['end_idx']
    
    # 4s窗口，如果行驶距离小于2米，基本是静止或蠕动，建议过滤掉以减少中心点噪声
    if win.get('dist', 0) < 2.0:
        continue
    
    if seq_id not in seq_to_frames: continue
    frames = seq_to_frames[seq_id][start_idx : end_idx]
    if len(frames) < 2: continue
        
    base_frame = frames[0]
    curr_x, curr_y, curr_yaw = base_frame['x'], base_frame['y'], base_frame['heading']
    cos_yaw, sin_yaw = np.cos(curr_yaw), np.sin(curr_yaw)

    for i in range(1, len(frames)):
        dx = frames[i]['x'] - curr_x
        dy = frames[i]['y'] - curr_y
        
        # 坐标转换
        local_forward = dx * cos_yaw + dy * sin_yaw
        local_right = dx * sin_yaw - dy * cos_yaw
        
        # 4s轨迹可能存在掉头，但大部分关注前向
        if local_forward > -1.0: 
            relative_coords.append([local_right, local_forward])

relative_coords = np.array(relative_coords)

# --- 绘图优化渲染 ---
print(f"\nGenerating 4s Balanced Heatmap...")
# 调整画布比例：由于纵向距离远大于横向，(10, 12) 或 (8, 12) 更有“道路感”
plt.figure(figsize=(10, 12))

# 核心参数调整：
# 1. gridsize 设为 120-150：点数多了，格子可以细一点，展现平滑的弯道轨迹
# 2. bins='log'：对于采样后的平衡数据，log能让转弯处（采样较少）也清晰可见
hb = plt.hexbin(
    relative_coords[:, 0], 
    relative_coords[:, 1], 
    gridsize=130,      
    cmap='YlOrRd', 
    bins='log', 
    mincnt=1,
    edgecolors='none'  # 去掉六边形边缘线，让颜色过渡更自然
)

# 装饰器
cb = plt.colorbar(hb, label='log10(frame count)')
plt.axvline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
plt.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)

# --- 关键：适配 4s 窗口的坐标范围 ---
# 侧向：4s 变道或转弯大约在 15-20m 左右
plt.xlim(-25, 25) 
# 前向：4s 行驶距离根据车速不同，通常在 0-100m 之间
plt.ylim(-2, 100) 

plt.xlabel('Lateral Distance (Right is +X) [m]', fontsize=12)
plt.ylabel('Forward Distance (Forward is +Y) [m]', fontsize=12)
plt.title('4.0s Balanced Trajectory Distribution (40 frames)\nBalanced Across Motion Angles', fontsize=14)

plt.grid(True, alpha=0.1, linestyle=':')
# 强制白底，并调整背景色
plt.gca().set_facecolor('#fdfdfd')

save_path = os.path.join(SAVE_DIR, "waymo_balanced_4s_heatmap.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Success! 4s heatmap saved to {save_path}")
print(f"Stats: Max Forward={np.max(relative_coords[:,1]):.1f}m, Max Lateral={np.max(np.abs(relative_coords[:,0])):.1f}m")
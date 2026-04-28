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

# 将原始帧按 seq_id 索引，方便快速查找
seq_to_frames = defaultdict(list)
for f in all_frames:
    seq_to_frames[f['seq_id']].append(f)

# 确保每个序列内的帧是按时间排序的
for seq_id in seq_to_frames:
    seq_to_frames[seq_id].sort(key=lambda x: x['timestamp_micros'])

relative_coords = []
direction_count = defaultdict(int)

print(f"Processing {len(balanced_windows)} balanced windows...")
for win_idx, win in enumerate(balanced_windows):
    seq_id = win['seq_id']
    start_idx = win['start_idx']
    end_idx = win['end_idx']
    
    if seq_id not in seq_to_frames:
        print(f"Warning: seq_id {seq_id} not found, skip window {win_idx}")
        continue
    frames = seq_to_frames[seq_id]
    if end_idx > len(frames):
        print(f"Warning: end_idx {end_idx} out of range for seq {seq_id}, skip window {win_idx}")
        continue
    
    frames_in_win = frames[start_idx : end_idx]
    if len(frames_in_win) < 2:
        continue
        
    base_frame = frames_in_win[0]
    curr_x, curr_y, curr_yaw = base_frame['x'], base_frame['y'], base_frame['heading']
    
    cos_yaw = np.cos(curr_yaw)
    sin_yaw = np.sin(curr_yaw)

    for i in range(1, len(frames_in_win)):
        fut = frames_in_win[i]
        dx = fut['x'] - curr_x
        dy = fut['y'] - curr_y
        
        # --- 恢复正确的坐标转换公式 ---
        local_forward = dx * cos_yaw + dy * sin_yaw
        local_right = dx * sin_yaw - dy * cos_yaw
        
        # 可选：过滤掉明显的倒车点（例如 forward < -1m）
        if local_forward < -1:
            continue
            
        relative_coords.append([local_right, local_forward])
        
        if abs(local_right) < 0.5:
            direction_count['straight'] += 1
        elif local_right > 0:
            direction_count['right'] += 1
        else:
            direction_count['left'] += 1

relative_coords = np.array(relative_coords)
print(f"\n=== Data Statistics ===")
print(f"Total relative points: {len(relative_coords)}")
print(f"Direction distribution: {dict(direction_count)}")
if len(relative_coords) > 0:
    print(f"Local right range: [{np.min(relative_coords[:,0]):.1f}, {np.max(relative_coords[:,0]):.1f}] m")
    print(f"Local forward range: [{np.min(relative_coords[:,1]):.1f}, {np.max(relative_coords[:,1]):.1f}] m")
    backward_ratio = np.sum(relative_coords[:,1] < 0) / len(relative_coords) * 100
    print(f"Backward points ratio: {backward_ratio:.2f}%")
else:
    print("Error: No valid relative points generated!")
    exit()

# 2. 绘制热力图
print(f"\nGenerating Heatmap...")
plt.figure(figsize=(10, 10))

hb = plt.hexbin(
    relative_coords[:, 0], 
    relative_coords[:, 1], 
    gridsize=50,
    cmap='YlOrRd', 
    bins='log',
    mincnt=1,
    alpha=0.8
)

cb = plt.colorbar(hb, label='log10(frame count)')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.xlabel('Lateral Distance (Left <--- 0 ---> Right) [m]')
plt.ylabel('Forward Distance [m]')
plt.title('Sampled Balanced Trajectory Heatmap (0.8-sec Windows, 8 frames)')

# 动态调整坐标范围，匹配数据分布
x_min, x_max = np.min(relative_coords[:,0]), np.max(relative_coords[:,0])
y_min, y_max = np.min(relative_coords[:,1]), np.max(relative_coords[:,1])
x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 1
y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 1
plt.xlim(x_min - x_pad, x_max + x_pad)
plt.ylim(y_min - y_pad, y_max + y_pad)

plt.grid(True, alpha=0.3)
plt.gca().set_facecolor('#f8f8f8')

save_path = os.path.join(SAVE_DIR, "waymo_balanced_trajectory_heatmap_corrected.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Success! Heatmap saved to {save_path}")
import json
import numpy as np
import os
import random
from collections import defaultdict
# ========== 新增：导入进度条库 ==========
from tqdm import tqdm

# 1. 配置
METADATA_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/prepro_waymo/waymo_ego_metadata.json"
SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/waymo_balanced"
WINDOW_SIZE = 40 # 假设 10Hz，2秒即20帧
STEP_SIZE = 20    # 步长，设为 WINDOW_SIZE 表示无重叠
NUM_BINS = 36     # 36个扇区，每个10度
SAMPLES_PER_BIN = 150 # 每个扇区期望采样的窗口数（根据你的数据量调整）

# 2. 加载元数据
with open(METADATA_PATH, 'r') as f:
    all_frames = json.load(f)

# 3. 按 sequence 分组
seq_groups = defaultdict(list)
for frame in all_frames:
    seq_groups[frame['seq_id']].append(frame)

# 4. 生成 Windows 并计算 Motion Angle
windows = []
print("Generating windows and calculating motion angles...")

# ========== 修改：给序列循环加进度条 ==========
for seq_id, frames in tqdm(seq_groups.items(), desc="Processing sequences"):
    # 确保帧按时间排序
    frames.sort(key=lambda x: x['timestamp_micros'])
    
    for i in range(0, len(frames) - WINDOW_SIZE, STEP_SIZE):
        start_f = frames[i]
        end_f = frames[i + WINDOW_SIZE - 1]
        
        # 计算窗口的总位移 (End - Start)
        dx = end_f['x'] - start_f['x']
        dy = end_f['y'] - start_f['y']
        
        dist = np.sqrt(dx**2 + dy**2)
        
        # 过滤掉几乎不动的窗口（静止时角度无意义）
        if dist < 1.0: # 2秒位移小于1米视为静止
            continue
            
        # 计算角度: θ = atan2(dx, dy) 
        # y轴正向(Forward)为0度
        angle = np.arctan2(dx, dy) 
        
        windows.append({
            "seq_id": seq_id,
            "start_idx": i,
            "end_idx": i + WINDOW_SIZE,
            "start_timestamp": start_f['timestamp_micros'],
            "angle": angle,
            "dist": dist
        })

print(f"Total valid windows generated: {len(windows)}")

with open(os.path.join(SAVE_DIR, "all_windows_metadata.json"), "w") as f:
    json.dump(windows, f)
# 5. 在角度空间分桶并均匀采样
bins = defaultdict(list)
bin_width = (2 * np.pi) / NUM_BINS

# ========== 修改：给窗口分桶循环加进度条 ==========
for w in tqdm(windows, desc="Binning windows"):
    # 将 angle [-pi, pi] 映射到 [0, 2*pi]
    theta = (w['angle'] + np.pi) % (2 * np.pi)
    bin_idx = int(theta // bin_width) % NUM_BINS
    bins[bin_idx].append(w)

# 执行采样
balanced_windows = []
print(f"Sampling {SAMPLES_PER_BIN} windows per bin...")

# ========== 修改：给扇区采样循环加进度条 ==========
for b_idx in tqdm(range(NUM_BINS), desc="Sampling bins"):
    curr_bin_windows = bins[b_idx]
    count = len(curr_bin_windows)
    
    if count == 0:
        print(f"Warning: Bin {b_idx} is empty.")
        continue
    
    # 如果该方向数据不足，则取全部；否则随机抽样
    num_to_sample = min(count, SAMPLES_PER_BIN)
    sampled = random.sample(curr_bin_windows, num_to_sample)
    balanced_windows.extend(sampled)
    
    print(f"Bin {b_idx} ({np.rad2deg(b_idx*bin_width - np.pi):.1f}°): Found {count}, Sampled {num_to_sample}")

# 6. 保存结果
output_path = os.path.join(SAVE_DIR, "balanced_windows_metadata.json")
with open(output_path, 'w') as f:
    json.dump(balanced_windows, f, indent=2)

print(f"\nFinal balanced dataset size: {len(balanced_windows)} windows.")
print(f"Saved to: {output_path}")

# 7. 可视化采样后的分布 (验证)
import matplotlib.pyplot as plt
sampled_angles = [w['angle'] for w in balanced_windows]
plt.figure(figsize=(10, 5))
plt.hist(sampled_angles, bins=NUM_BINS, range=(-np.pi, np.pi), rwidth=0.8)
plt.title("Balanced Window Distribution by Motion Angle")
plt.xlabel("Angle (rad)")
plt.ylabel("Window Count")
plt.savefig(os.path.join(SAVE_DIR, "balanced_distribution_check.png"))

# ========= 极坐标对比图 =========
print("Drawing polar comparison...")

all_angles = [w['angle'] for w in windows]
balanced_angles = [w['angle'] for w in balanced_windows]

bin_edges = np.linspace(-np.pi, np.pi, NUM_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
width = (2 * np.pi) / NUM_BINS

all_counts, _ = np.histogram(all_angles, bins=bin_edges)
balanced_counts, _ = np.histogram(balanced_angles, bins=bin_edges)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# 筛选前（灰色）
ax.bar(
    bin_centers,
    all_counts,
    width=width,
    color="lightgray",
    edgecolor="gray",
    alpha=0.6,
    label="Before balancing"
)

# 筛选后（红色）
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

ax.set_title("Motion Angle Distribution (Polar View)", pad=20)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "polar_distribution_comparison.png"), dpi=300)
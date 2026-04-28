import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ==============================
# 路径
# ==============================
DATA_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_val_sort/classified_val_windows.json"
SAVE_DIR = os.path.dirname(DATA_PATH)

with open(DATA_PATH, "r") as f:
    windows = json.load(f)

print(f"Loaded {len(windows)} windows")

# ==============================
# 组织数据
# ==============================
labels = []
cum_yaw = []
lateral = []
v_mean = []
max_acc = []
max_dec = []

for w in windows:
    labels.append(w["label"])
    m = w["metrics"]
    cum_yaw.append(m["cum_yaw_deg"])
    lateral.append(m["lateral_shift_m"])
    v_mean.append(m["v_mean_m_s"])
    max_acc.append(m["max_acc_m_s2"])
    max_dec.append(m["max_dec_m_s2"])

labels = np.array(labels)
cum_yaw = np.array(cum_yaw)
lateral = np.array(lateral)
v_mean = np.array(v_mean)
max_acc = np.array(max_acc)
max_dec = np.array(max_dec)

# 固定类别顺序（更专业）
class_order = ["idle", "straight", "lane_change", "turning", "aggressive"]

# 固定颜色
color_map = {
    "idle": "#7f7f7f",        # gray
    "straight": "#2ca02c",    # green
    "lane_change": "#1f77b4", # blue
    "turning": "#ff7f0e",     # orange
    "aggressive": "#d62728"   # red
}

# ====================================================
# 1️⃣ cum_yaw vs lateral_shift
# ====================================================
plt.figure(figsize=(8,6))

for lab in class_order:
    idx = labels == lab
    if np.sum(idx) == 0:
        continue
    plt.scatter(
        cum_yaw[idx],
        lateral[idx],
        s=22,
        alpha=0.65,
        c=color_map[lab],
        label=f"{lab} ({np.sum(idx)})"
    )

plt.xlabel("Cumulative Yaw Change (deg)")
plt.ylabel("Lateral Shift (m)")
plt.title("Yaw vs Lateral Shift")
plt.legend()
plt.grid(True)
plt.tight_layout()

out1 = os.path.join(SAVE_DIR, "scatter_yaw_vs_lateral.png")
plt.savefig(out1, dpi=300)
print("Saved:", out1)

# ====================================================
# 2️⃣ 速度 vs 最大加速度
# ====================================================
plt.figure(figsize=(8,6))

for lab in class_order:
    idx = labels == lab
    if np.sum(idx) == 0:
        continue
    plt.scatter(
        v_mean[idx],
        max_acc[idx],
        s=22,
        alpha=0.65,
        c=color_map[lab],
        label=f"{lab} ({np.sum(idx)})"
    )

plt.xlabel("Mean Speed (m/s)")
plt.ylabel("Max Acceleration (m/s²)")
plt.title("Speed vs Acceleration")
plt.legend()
plt.grid(True)
plt.tight_layout()

out2 = os.path.join(SAVE_DIR, "scatter_speed_vs_acc.png")
plt.savefig(out2, dpi=300)
print("Saved:", out2)

# ====================================================
# 3️⃣ 类别数量柱状图（颜色统一）
# ====================================================
count_dict = Counter(labels)
counts = [count_dict[c] for c in class_order]

plt.figure(figsize=(7,5))
plt.bar(class_order, counts, color=[color_map[c] for c in class_order])

plt.title("Window Count per Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()

out3 = os.path.join(SAVE_DIR, "class_distribution.png")
plt.savefig(out3, dpi=300)
print("Saved:", out3)

print("\nVisualization complete.")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# 1. 路径配置
METADATA_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/prepro_waymo/waymo_ego_metadata.json"
SAVE_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh"

print("Loading metadata...")
with open(METADATA_PATH, 'r') as f:
    all_ego = json.load(f)

# 2. 提取位移 (dx, dy) 
# 注意：必须在同一个 seq_id 内计算位移，避免跳变
dx_list = []
dy_list = []

print("Calculating displacements...")
for i in range(1, len(all_ego)):
    if all_ego[i]['seq_id'] == all_ego[i-1]['seq_id']:
        # 获取增量
        dx = all_ego[i]['x'] - all_ego[i-1]['x']
        dy = all_ego[i]['y'] - all_ego[i-1]['y']
        
        # 过滤静止帧 (位移太小会导致角度噪声极大)
        if np.sqrt(dx**2 + dy**2) > 0.05: 
            dx_list.append(dx)
            dy_list.append(dy)

dx = np.array(dx_list)
dy = np.array(dy_list)

# 3. 极坐标转换
# 按照你的定义：Forward 是 y, Lateral 是 x
# θ = atan2(x, y) -> y正半轴为0度，x正半轴为90度
# r = sqrt(x^2 + y^2)
r = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(dx, dy) # 返回 [-pi, pi]

# 4. 扇区统计与逆频率加权
num_bins = 36
# 将 theta 映射到 [0, 2*pi] 方便分桶
theta_mapped = (theta + np.pi) % (2 * np.pi)
counts, bin_edges = np.histogram(theta_mapped, bins=num_bins, range=(0, 2*np.pi))

# 计算逆频率权重: weight = 1 / frequency
# 加入 small epsilon 防止除零
weights = 1.0 / (counts + 1e-6)
# 归一化权重，防止数值过大
weights = weights / np.sum(weights)

# 5. 准备可视化数据
# 极坐标图的 X 是角度，Y 是权重（或加权后的计数）
bin_centers = (bin_edges[:-1] + bin_edges[1:]) - np.pi # 转换回 [-pi, pi] 中心

# 6. 绘制第一个图：极坐标权重分布柱状图
print("Plotting Polar Weighted Distribution...")
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

# 设置极坐标 0 度朝上 (N)
ax.set_theta_zero_location('N')
# 设置顺时针方向或逆时针 (Waymo 通常 x 右 y 前，即顺时针)
ax.set_theta_direction(-1) 

# 绘制柱状图展示权重分布
bars = ax.bar(bin_centers, weights, width=2*np.pi/num_bins, bottom=0.0, 
              color=plt.cm.viridis(weights / np.max(weights)), edgecolor='k', alpha=0.8)

plt.title("Polar Inverse Frequency Weighting (Directional Distribution)", va='bottom', fontsize=15)
ax.set_xticklabels(['Forward', '45°', 'Right', '135°', 'Backward', '-135°', 'Left', '-45°'])

# 保存第一个图
save_path = os.path.join(SAVE_DIR, "waymo_polar_weighted_dist.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Polar plot saved to {save_path}")

# 7. 绘制第二个图：r-theta 散点热力图（修正 ax2 定义问题）
print("Plotting Polar Heatmap (R vs Theta)...")
plt.figure(figsize=(10, 8))
# 关键：正确定义 ax2 变量
ax2 = plt.subplot(111, projection='polar')
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)

# 只看前向 (过滤 r <= 2 的数据，限制显示范围在 2 米以内)
mask = r <= 2
theta_filtered = theta[mask]
r_filtered = r[mask]

# 绘制散点图，点大小调大到50（更明显）
sc = ax2.scatter(
    theta_filtered, r_filtered,
    s=20,          # 固定点大小，不再和 r 绑定
    c=r_filtered,  # 颜色表示距离
    cmap='magma',
    alpha=0.4,     # 降低透明度，避免重叠
    edgecolors='none'
)
# 只保留一个colorbar，对应散点图
plt.colorbar(sc, label='r (distance, m)')
# 限制r轴显示范围，聚焦前向2米
ax2.set_ylim(0, 2)
plt.title("Motion Intensity (r vs θ)")

# 保存第二个图
heatmap_save_path = os.path.join(SAVE_DIR, "waymo_polar_heatmap.png")
plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Heatmap plot saved to {heatmap_save_path}")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

ply_path = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/output/debug_plucker/ray_debug/ray_frustum_b0_t0_all_views.ply"
png_path = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/output/debug_plucker/ray_debug/ray_frustum_b0_t0_all_views.png"

with open(ply_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

end_header_idx = 0
vertex_count = 0

for i, line in enumerate(lines):
    if line.startswith("element vertex"):
        vertex_count = int(line.strip().split()[-1])
    if line.strip() == "end_header":
        end_header_idx = i
        break

data_lines = lines[end_header_idx + 1:end_header_idx + 1 + vertex_count]

xyz = []
rgb = []

for line in data_lines:
    parts = line.strip().split()
    if len(parts) < 6:
        continue
    x = float(parts[0])
    y = float(parts[1])
    z = float(parts[2])
    r = int(parts[3])
    g = int(parts[4])
    b = int(parts[5])
    xyz.append([x, y, z])
    rgb.append([r, g, b])

xyz = np.array(xyz, dtype=np.float32)
rgb = np.array(rgb, dtype=np.float32) / 255.0

print("num_points =", xyz.shape[0])
print("xyz min =", xyz.min(axis=0))
print("xyz max =", xyz.max(axis=0))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

max_points = 30000
if xyz.shape[0] > max_points:
    step = int(np.ceil(xyz.shape[0] / max_points))
    xyz = xyz[::step]
    rgb = rgb[::step]

ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    c=rgb,
    s=1,
    depthshade=False,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("ray_frustum_b0_t0_all_views")

x_mid = (xyz[:, 0].max() + xyz[:, 0].min()) * 0.5
y_mid = (xyz[:, 1].max() + xyz[:, 1].min()) * 0.5
z_mid = (xyz[:, 2].max() + xyz[:, 2].min()) * 0.5

x_range = xyz[:, 0].max() - xyz[:, 0].min()
y_range = xyz[:, 1].max() - xyz[:, 1].min()
z_range = xyz[:, 2].max() - xyz[:, 2].min()
half = max(x_range, y_range, z_range) * 0.5

ax.set_xlim(x_mid - half, x_mid + half)
ax.set_ylim(y_mid - half, y_mid + half)
ax.set_zlim(z_mid - half, z_mid + half)

ax.view_init(elev=22, azim=-58)

plt.tight_layout()
plt.savefig(png_path, dpi=220)
print("saved to:", png_path)
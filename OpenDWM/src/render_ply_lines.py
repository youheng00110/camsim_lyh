import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


parser = argparse.ArgumentParser()
parser.add_argument("--ply", required=True)
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

with open(args.ply, "r") as f:
    lines = f.readlines()

vertex_count = 0
edge_count = 0
header_end = 0

for line_id, line in enumerate(lines):
    line = line.strip()

    if line.startswith("element vertex"):
        vertex_count = int(line.split()[-1])

    if line.startswith("element edge"):
        edge_count = int(line.split()[-1])

    if line == "end_header":
        header_end = line_id + 1
        break

vertex_lines = lines[header_end:header_end + vertex_count]
edge_lines = lines[header_end + vertex_count:header_end + vertex_count + edge_count]

vertices = []
for line in vertex_lines:
    parts = line.strip().split()
    x = float(parts[0])
    y = float(parts[1])
    z = float(parts[2])
    r = int(parts[3])
    g = int(parts[4])
    b = int(parts[5])
    vertices.append((x, y, z, r, g, b))

edges = []
for line in edge_lines:
    parts = line.strip().split()
    v0 = int(parts[0])
    v1 = int(parts[1])
    r = int(parts[2])
    g = int(parts[3])
    b = int(parts[4])
    edges.append((v0, v1, r, g, b))

xs = [v[0] for v in vertices]
ys = [v[1] for v in vertices]
zs = [v[2] for v in vertices]

x_mid = (min(xs) + max(xs)) * 0.5
y_mid = (min(ys) + max(ys)) * 0.5
z_mid = (min(zs) + max(zs)) * 0.5

max_range = max(
    max(xs) - min(xs),
    max(ys) - min(ys),
    max(zs) - min(zs),
) * 0.55

views = [
    ("iso", 25, -45),
    ("top", 90, -90),
    ("front", 0, -90),
    ("side", 0, 0),
]

for name, elev, azim in views:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for v0, v1, r, g, b in edges:
        p0 = vertices[v0]
        p1 = vertices[v1]
        color = (r / 255.0, g / 255.0, b / 255.0)

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=0.6,
            color=color,
        )

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(name)

    output_path = os.path.join(args.out_dir, "{}.png".format(name))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

print("saved to:", args.out_dir)

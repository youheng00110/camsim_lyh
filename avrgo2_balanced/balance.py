import os
import json
import numpy as np
import random
import pandas as pd
import glob

from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


# ===============================
# 1️⃣ 配置
# ===============================

JSON_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/avrgo2_json"
DATA_ROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/Avrgoverse2_sensor_data"

SAVE_DIR = "./avrgo2_balanced"

WINDOW_SEC = 4
STEP_SEC = 1

CAMERA_NAME = "ring_front_center"

NUM_BINS = 36
SAMPLES_PER_BIN = 150

os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# 2️⃣ 读取 scene
# ===============================

seq_frames = defaultdict(list)

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

print("Loading scenes...")

for jf in tqdm(json_files):

    with open(os.path.join(JSON_DIR, jf)) as f:
        meta = json.load(f)

    scene = meta["scene_name"]
    path = meta["path"]

    scene_root = os.path.join(DATA_ROOT, path)

    ego_file = os.path.join(scene_root, "city_SE3_egovehicle.feather")

    if not os.path.exists(ego_file):
        continue

    # ===============================
    # ego pose
    # ===============================

    df = pd.read_feather(ego_file)

    ego_ts = df["timestamp_ns"].values
    ego_x = df["tx_m"].values
    ego_y = df["ty_m"].values

    # ===============================
    # camera timestamps
    # ===============================

    cam_dir = os.path.join(scene_root,
        f"sensors/cameras/{CAMERA_NAME}")

    imgs = sorted(glob.glob(cam_dir+"/*.jpg"))

    cam_ts = np.array([
        int(os.path.basename(i).split(".")[0]) for i in imgs
    ])

    # ===============================
    # 最近邻匹配 ego
    # ===============================

    idx = np.searchsorted(ego_ts, cam_ts)

    idx = np.clip(idx,1,len(ego_ts)-1)

    left = ego_ts[idx-1]
    right = ego_ts[idx]

    idx -= cam_ts-left < right-cam_ts

    xs = ego_x[idx]
    ys = ego_y[idx]

    # ===============================
    # 保存 frame
    # ===============================

    for t,x,y in zip(cam_ts,xs,ys):

        seq_frames[scene].append({
            "timestamp":int(t),
            "x":float(x),
            "y":float(y)
        })


print("Total scenes:",len(seq_frames))


# ===============================
# 3️⃣ camera fps
# ===============================

sample = next(iter(seq_frames.values()))

ts = np.array([f["timestamp"] for f in sample])

dt = np.median(np.diff(ts))/1e9
fps = 1/dt

print("Camera FPS:",fps)

WINDOW_SIZE = int(WINDOW_SEC*fps)
STEP_SIZE = int(STEP_SEC*fps)

print("Window size:",WINDOW_SIZE)
print("Step size:",STEP_SIZE)


# ===============================
# 4️⃣ sliding windows
# ===============================

windows = []

print("Generating windows...")

for scene, frames in tqdm(seq_frames.items()):

    frames.sort(key=lambda x: x["timestamp"])

    for i in range(0, len(frames) - WINDOW_SIZE, STEP_SIZE):

        start = frames[i]
        end = frames[i + WINDOW_SIZE - 1]

        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]

        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1.0:
            continue

        angle = np.arctan2(dx, dy)

        windows.append({

            "scene_name": scene,
            "start_timestamp": start["timestamp"],
            "end_timestamp": end["timestamp"],
            "angle": float(angle),
            "dist": float(dist)

        })

print("Total windows before balance:", len(windows))


# ===============================
# 保存所有 windows
# ===============================

all_json = os.path.join(SAVE_DIR, "all_windows.json")

with open(all_json, "w") as f:
    json.dump(windows, f, indent=2)

print("Saved all windows:", all_json)

# ===============================
# 5️⃣ 分桶
# ===============================

bins=defaultdict(list)

bin_width=(2*np.pi)/NUM_BINS

for w in windows:

    theta=(w["angle"]+np.pi)%(2*np.pi)

    idx=int(theta//bin_width)%NUM_BINS

    bins[idx].append(w)


# ===============================
# 6️⃣ 采样
# ===============================

balanced=[]

print("Sampling bins...")

for b in range(NUM_BINS):

    curr=bins[b]

    if len(curr)==0:
        continue

    n=min(len(curr),SAMPLES_PER_BIN)

    sampled=random.sample(curr,n)

    balanced.extend(sampled)

    print(f"bin {b}: {len(curr)} -> {n}")


print("Final windows after balance:",len(balanced))


# ===============================
# 7️⃣ 保存
# ===============================

out_json=os.path.join(SAVE_DIR,"balanced_windows.json")

with open(out_json,"w") as f:

    json.dump(balanced,f,indent=2)

print("Saved:",out_json)


# ===============================
# 8️⃣ 可视化
# ===============================

all_angles=[w["angle"] for w in windows]
bal_angles=[w["angle"] for w in balanced]

bin_edges=np.linspace(-np.pi,np.pi,NUM_BINS+1)

all_counts,_=np.histogram(all_angles,bins=bin_edges)
bal_counts,_=np.histogram(bal_angles,bins=bin_edges)

bin_centers=(bin_edges[:-1]+bin_edges[1:])/2
width=(2*np.pi)/NUM_BINS

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111,polar=True)

ax.bar(bin_centers,all_counts,width=width,color="lightgray",alpha=0.6,label="Before")
ax.bar(bin_centers,bal_counts,width=width,color="red",alpha=0.7,label="After")

ax.set_ylim(0,200)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.legend()

plt.savefig(os.path.join(SAVE_DIR,"polar_compare.png"),dpi=300)

print("Polar plot saved.")
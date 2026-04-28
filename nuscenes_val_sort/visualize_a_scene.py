import os
import cv2
from nuscenes import NuScenes

# ===============================
# 1️⃣ 路径
# ===============================
DATAROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_link"
SAVE_VIDEO = "scene_0926_turning.mp4"

# ===============================
# 2️⃣ 你的 sample tokens
# ===============================
sample_tokens = [
    "45eda670b5a840acbb730aac15e63b19",
      "25c6e881c42c4b1990449c661ae6532c",
      "7bb8d00122624267a777a5ed487b8703",
      "48ba767f9b8248a5b6f6921182e053b4",
      "78989332f49b47ada534ecfe51c78dd5",
      "7c4bfd7aa61f4fb4a0e87d82a0b9609e",
      "3f2adb9db8ab428abd54009707d46992",
      "c06a5e8ca3694889a25d3143d4dca9d5",
      "6fbe3d91fb7944b798279abcc69651cb",
      "d11e40fecf3748db910241be9fc0d86b",
      "8a05f7a2995b44beadcff8d41caef34b",
      "6d24000d98854d99b5af251bfdf31561",
      "4294b2841a7e4931a4504a98ffc31dfc",
      "7a7073a4f4d74a0a8b90150dd5776e7a",
      "26d153ce37804d82822823a6c5b9943a",
      "91a161db9a194e2d8473a0d86793da53",
      "a522481fa0b8439eb27208e20a914c78",
      "6f63778e796b4770bfbd5d347f5ef485",
      "4d5670f980de423cb8ec0c656349a648",
      "1998255a63a7449aaf471c86905f2199"
]

# ===============================
# 3️⃣ 加载 nuScenes
# ===============================
print("Loading nuScenes...")
nusc = NuScenes(version="v1.0-trainval",
                dataroot=DATAROOT,
                verbose=False)

# ===============================
# 4️⃣ 读取第一帧确定视频尺寸
# ===============================
first_sample = nusc.get("sample", sample_tokens[0])
cam_token = first_sample["data"]["CAM_FRONT"]
cam_data = nusc.get("sample_data", cam_token)

img_path = os.path.join(nusc.dataroot, cam_data["filename"])
frame = cv2.imread(img_path)

height, width, _ = frame.shape

# ===============================
# 5️⃣ 创建视频写入器
# ===============================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(SAVE_VIDEO, fourcc, 2, (width, height))
# 2 FPS = 10秒窗口正好播放完

# ===============================
# 6️⃣ 写入所有帧
# ===============================
for token in sample_tokens:
    sample = nusc.get("sample", token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)

    img_path = os.path.join(nusc.dataroot, cam_data["filename"])
    frame = cv2.imread(img_path)

    video.write(frame)

video.release()

print("Saved video to:", SAVE_VIDEO)
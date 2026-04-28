import os
import sys
import pickle
import json
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import multiprocessing

# 1. 路径配置（保持你的正确路径）
PROTO_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/externals/waymo-open-dataset/src"
INPUT_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/waymo_link/individual_files/training"
OUTPUT_DIR = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/prepro_waymo"

if PROTO_PATH not in sys.path:
    sys.path.append(PROTO_PATH)

from waymo_open_dataset import dataset_pb2

def process_single_tfrecord(file_name):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_name = file_name.replace(".tfrecord", "_ego.pkl")
    output_path = os.path.join(OUTPUT_DIR, output_name)

    if os.path.exists(output_path):
        return None

    ego_info_list = []
    
    # 用于手动计算速度和曲率的缓存变量
    last_pos = None      # (x, y)
    last_timestamp = None
    last_heading = None

    try:
        dataset = tf.data.TFRecordDataset(input_path, compression_type='')
        
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            # --- 1. 提取基础位姿 (4x4 矩阵) ---
            tr = frame.pose.transform
            curr_x, curr_y, curr_z = tr[3], tr[7], tr[11]
            curr_heading = math.atan2(tr[4], tr[0]) # 从旋转矩阵计算 Yaw
            curr_ts = frame.timestamp_micros / 1e6  # 转为秒
            
            # --- 2. 手动计算速度和曲率 ---
            speed = 0.0
            curvature = 0.0
            
            if last_pos is not None:
                dt = curr_ts - last_timestamp
                if dt > 0:
                    # 距离差
                    dist = math.sqrt((curr_x - last_pos[0])**2 + (curr_y - last_pos[1])**2)
                    # 速度 = 距离 / 时间
                    speed = dist / dt
                    
                    # 曲率 = 航向角变化量 / 距离
                    # 使用 atan2 处理角度跨越 -pi 到 pi 的情况
                    d_heading = (curr_heading - last_heading + math.pi) % (2 * math.pi) - math.pi
                    if dist > 0.1: # 只有车辆移动了，计算曲率才有意义
                        curvature = d_heading / dist

            # 更新缓存
            last_pos = (curr_x, curr_y)
            last_timestamp = curr_ts
            last_heading = curr_heading

            # --- 3. 存储结果 ---
            ego_info = {
                "seq_id": frame.context.name,
                "frame_id": frame.timestamp_micros,
                "timestamp_micros": frame.timestamp_micros,
                "x": curr_x,
                "y": curr_y,
                "z": curr_z,
                "heading": curr_heading,
                "speed": speed,
                "curvature": curvature,
                "image_names": [cam.name for cam in frame.images]
            }
            ego_info_list.append(ego_info)
            
        with open(output_path, 'wb') as f:
            pickle.dump(ego_info_list, f)
        return f"Success: {file_name}"

    except Exception as e:
        return f"Error: {file_name}, reason: {str(e)}"

def main():
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".tfrecord")])
    print(f"Total files: {len(all_files)}. Processing with 20 cores...")

    with multiprocessing.Pool(20) as pool:
        list(tqdm(pool.imap(process_single_tfrecord, all_files), total=len(all_files)))

    # 汇总
    print("Merging results...")
    all_data = []
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith("_ego.pkl"):
            with open(os.path.join(OUTPUT_DIR, f), 'rb') as pkl:
                all_data.extend(pickle.load(pkl))
    
    with open(os.path.join(OUTPUT_DIR, "waymo_ego_metadata.json"), 'w') as jf:
        json.dump(all_data, jf)
    print(f"Final metadata saved. Total frames: {len(all_data)}")

if __name__ == "__main__":
    main()
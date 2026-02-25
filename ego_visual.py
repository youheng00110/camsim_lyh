import sys
import tensorflow as tf

# 加入你自己编译的 proto 路径
sys.path.append("/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/externals/waymo-open-dataset/src")

from waymo_open_dataset import dataset_pb2

filename = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/prepro_waymo/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord"

dataset = tf.data.TFRecordDataset(filename)


for data in dataset.take(1):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    print("timestamp:", frame.timestamp_micros)
    print("num camera images:", len(frame.images))
    print("num 3D labels:", len(frame.laser_labels))
    pose = frame.pose.transform
    print("ego pose (flatten 4x4):", pose)

    tx, ty, tz = pose[3], pose[7], pose[11]
    print("ego position:", tx, ty, tz)
import dwm.common
import dwm.datasets.common
import dwm.datasets.waymo_common as wc
import fsspec
import io
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import transforms3d
import os
import waymo_open_dataset.dataset_pb2 as waymo_pb
import zlib


class MotionDataset(torch.utils.data.Dataset):
    """The motion data loaded from the Waymo Perception dataset.

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the data records.
        info_dict_path (str): The path to the info dict file, which contains
            the offset of the data at each timestamp in the record relative to
            the beginning of the file, is used for fast seek during random
            access.
        sequence_length (int): The frame count of the temporal sequence.
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, stride is the begin time in second
            between 2 adjacent video clips, else the stride is the index count
            of the beginning between 2 adjacent video clips.
        sensor_channels (list): The string list of required views in
            "LIDAR_TOP", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_SIDE_LEFT", "CAM_SIDE_RIGHT", following the Waymo sensor name.
        enable_camera_transforms (bool): If set to True, the data item will
            include the "camera_transforms", "camera_intrinsics", "image_size"
            if camera modality exists, and include "lidar_transforms" if LiDAR
            modality exists. For a detailed definition of transforms, please
            refer to the dataset README.
        enable_ego_transforms (bool): If set to True, the data item will
            include the "ego_transforms". For a detailed definition of
            transforms, please refer to the dataset README.
        _3dbox_image_settings (dict or None): If set, the data item will
            include the "3dbox_images".
        hdmap_image_settings (dict or None): If set, the data item will include
            the "hdmap_images".
        _3dbox_bev_settings (dict or None): If set, the data item will include
            the "3dbox_bev_images".
        hdmap_bev_settings (dict or None): If set, the data item will include
            the "hdmap_bev_images".
        image_description_settings (dict or None): If set, the data item will
            include the "image_description". The "path" in the setting is for
            the content JSON file. The "time_list_dict_path" in the setting is
            for the file to seek the nearest labelled time points. Please refer
            to dwm.datasets.common.make_image_description_string() for details.
        stub_key_data_dict (dict or None): The dict of stub key and data, to
            align with other datasets with keys and data missing in this
            dataset. Please refer to dwm.datasets.common.add_stub_key_data()
            for details.
    """

    sensor_name_id_dict = {
        "CAM_FRONT": 1,
        "CAM_FRONT_LEFT": 2,
        "CAM_FRONT_RIGHT": 3,
        "CAM_SIDE_LEFT": 4,
        "CAM_SIDE_RIGHT": 5,
        "LIDAR_TOP": 1,
        "LIDAR_FRONT": 2,
        "LIDAR_SIDE_LEFT": 3,
        "LIDAR_SIDE_RIGHT": 4,
        "LIDAR_REAR": 5
    }
    box_type_dict = {
        "VEHICLE": 1,
        "PEDESTRIAN": 2,
        "SIGN": 3,
        "CYCLIST": 4
    }
    map_element_type_dict = {
        "road_line": "polyline",
        "lane": "polyline",
        "road_edge": "polyline",
        "crosswalk": "polygon",
        "driveway": "polygon",
        "speed_bump": "polygon"
    }

    extrinsic_correction = [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    default_3dbox_color_table = {
        "PEDESTRIAN": (255, 0, 0),
        "CYCLIST": (0, 255, 0),
        "VEHICLE": (0, 0, 255)
    }
    default_hdmap_color_table = {
        "crosswalk": (255, 0, 0),
        "road_edge": (0, 0, 255),
        "road_line": (0, 255, 0)
    }
    default_3dbox_corner_template = [
        [-0.5, -0.5, -0.5, 1], [-0.5, -0.5, 0.5, 1],
        [-0.5, 0.5, -0.5, 1], [-0.5, 0.5, 0.5, 1],
        [0.5, -0.5, -0.5, 1], [0.5, -0.5, 0.5, 1],
        [0.5, 0.5, -0.5, 1], [0.5, 0.5, 0.5, 1]
    ]
    default_3dbox_edge_indices = [
        (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        (6, 3), (6, 5)
    ]
    default_bev_from_ego_transform = [
        [6.4, 0, 0, 320],
        [0, -6.4, 0, 320],
        [0, 0, -6.4, 0],
        [0, 0, 0, 1]
    ]
    default_bev_3dbox_corner_template = [
        [-0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1],
        [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1]
    ]
    default_bev_3dbox_edge_indices = [(0, 2), (2, 3), (3, 1), (1, 0)]

    @staticmethod
    def find_by_name(list_to_search, queried_name):
            for item in list_to_search:
                # 这里的 int() 强制转换能解决枚举对比问题
                if int(item.name) == int(queried_name):
                    return item
            return None

    @staticmethod
    def enumerate_segments(
        sample_list: list, sequence_length: int, fps, stride
    ):
        # enumerate segments for each scene
        timestamps = [i[0] for i in sample_list]
        if fps == 0:
            # frames are extracted by the index.
            stop = len(timestamps) - sequence_length + 1
            for t in range(0, stop, max(1, stride)):
                yield timestamps[t:t+sequence_length]

        else:
            # frames are extracted by the timestamp.
            def enumerate_begin_time(timestamps, sequence_duration, stride):
                s = timestamps[-1] / 1000000 - sequence_duration
                t = timestamps[0] / 1000000
                while t <= s:
                    yield t
                    t += stride

            for t in enumerate_begin_time(
                timestamps, sequence_length / fps, stride
            ):
                candidates = [
                    dwm.datasets.common.find_nearest(
                        timestamps, (t + i / fps) * 1000000, return_item=True)
                    for i in range(sequence_length)
                ]
                yield candidates

    @staticmethod
    def get_images_and_lidar_points(
        sensor_channels: list, frame: waymo_pb.Frame
    ):
        images = []
        lidar_points = []
        for i in sensor_channels:
            if i.startswith("LIDAR"):
                laser = MotionDataset.find_by_name(
                    frame.lasers, MotionDataset.sensor_name_id_dict[i])
                range_image = waymo_pb.MatrixFloat()
                range_image.ParseFromString(
                    zlib.decompress(
                        laser.ri_return1.range_image_compressed))
                range_image = np.array(range_image.data, np.float32)\
                    .reshape(range_image.shape.dims)

                laser_calibration = wc.laser_calibration_to_dict(
                    MotionDataset.find_by_name(
                        frame.context.laser_calibrations,
                        MotionDataset.sensor_name_id_dict[i]))

                if i == "LIDAR_TOP":
                    range_image_top_pose = waymo_pb.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        zlib.decompress(
                            laser.ri_return1.range_image_pose_compressed))
                    range_image_top_pose = np\
                        .array(range_image_top_pose.data, np.float32)\
                        .reshape(range_image_top_pose.shape.dims)
                    frame_pose = np.array(frame.pose.transform, np.float32)\
                        .reshape(4, 4)
                else:
                    range_image_top_pose = None
                    frame_pose = None

                lidar_points.append(
                    torch.tensor(
                        wc.convert_range_image_to_cartesian(
                            range_image, laser_calibration,
                            range_image_top_pose, frame_pose),
                        dtype=torch.float32))

            elif i.startswith("CAM"):
                frame_image = MotionDataset.find_by_name(
                    frame.images, MotionDataset.sensor_name_id_dict[i])
                with io.BytesIO(frame_image.image) as f:
                    image = Image.open(f)
                    image.load()

                images.append(image)

        return images, lidar_points

    @staticmethod
    def get_3dbox_image(
        laser_labels, camera_calibration, _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 8)
        color_table = _3dbox_image_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        native_color_table = {
            MotionDataset.box_type_dict[k]: v for k, v in color_table.items()
        }

        corner_templates = _3dbox_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", MotionDataset.default_3dbox_edge_indices)

        # get the transform from the ego space to the image space
        image_size = (camera_calibration.width, camera_calibration.height)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4])
        ec = np.array(MotionDataset.extrinsic_correction)
        ego_from_camera = np.array(
            camera_calibration.extrinsic.transform).reshape(4, 4)
        image_from_ego = intrinsic @ ec @ np.linalg.inv(ego_from_camera)

        # draw annotations to the image
        def list_annotation():
            for i in laser_labels:
                yield i

        def get_world_transform(i):
            scale = np.diag([i.box.length, i.box.width, i.box.height, 1])
            ego_from_annotation = dwm.datasets.common.get_transform(
                transforms3d.euler.euler2quat(
                    0, 0, i.box.heading).tolist(),
                [i.box.center_x, i.box.center_y, i.box.center_z])
            return ego_from_annotation @ scale

        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)
        dwm.datasets.common.draw_3dbox_image(
            draw, image_from_ego, list_annotation, get_world_transform,
            lambda i: i.type, pen_width, native_color_table, corner_templates,
            edge_indices)

        return image

    @staticmethod
    def draw_polygon_to_image(
        polygon: list, draw: ImageDraw, transform: np.array,
        max_distance: float, pen_color: tuple, pen_width: int
    ):
        if len(polygon) == 0:
            return

        polygon_nodes = np.array([
            [i[0], i[1], i[2], 1] for i in polygon
        ], np.float32).transpose()
        p = transform @ polygon_nodes
        m = len(polygon)
        for i in range(m):
            xy = dwm.datasets.common.project_line(
                p[:, i], p[:, (i + 1) % m], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def draw_line_to_image(
        line: list, draw: ImageDraw, transform: np.array, max_distance: float,
        pen_color: tuple, pen_width: int
    ):
        if len(line) == 0:
            return

        line_nodes = np.array([
            [i[0], i[1], i[2], 1] for i in line
        ], np.float32).transpose()
        p = transform @ line_nodes
        for i in range(1, len(line)):
            xy = dwm.datasets.common.project_line(
                p[:, i - 1], p[:, i], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def get_hdmap_image(
        map_features, camera_calibration, pose, hdmap_image_settings: dict
    ):
        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0
        pen_width = hdmap_image_settings["pen_width"] \
            if "pen_width" in hdmap_image_settings else 8
        color_table = hdmap_image_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)
        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0

        # get the transform from the world space to the image space
        image_size = (camera_calibration.width, camera_calibration.height)
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4])
        ec = np.array(MotionDataset.extrinsic_correction, np.float32)
        ego_from_camera = np.array(
            camera_calibration.extrinsic.transform, np.float32).reshape(4, 4)
        world_from_ego = np.array(pose.transform, np.float32).reshape(4, 4)
        image_from_world = intrinsic @ ec @ \
            np.linalg.inv(world_from_ego @ ego_from_camera)

        # draw annotations to the image
        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)

        type_polygons = {}
        type_polylines = {}
        for feat in map_features:
            type_ = feat.WhichOneof('feature_data')
            if (
                type_ not in color_table or
                type_ not in MotionDataset.map_element_type_dict
            ):
                continue

            type_poly = MotionDataset.map_element_type_dict[type_]
            items = getattr(getattr(feat, type_), type_poly)
            coors_3d = []
            for item in items:
                coors_3d.append([item.x, item.y, item.z])
            if len(coors_3d) > 0:
                if type_poly == "polyline":
                    if type_ not in type_polylines:
                        type_polylines[type_] = []
                    type_polylines[type_].append(coors_3d)
                else:
                    if type_ not in type_polygons:
                        type_polygons[type_] = []
                    type_polygons[type_].append(coors_3d)

        for k, v in type_polygons.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_polygon_to_image(
                        i, draw, image_from_world, max_distance, c, pen_width)

        for k, v in type_polylines.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_line_to_image(
                        i, draw, image_from_world, max_distance, c, pen_width)

        return image

    @staticmethod
    def get_3dbox_bev_image(laser_labels, _3dbox_bev_settings: dict):
        # options
        pen_width = _3dbox_bev_settings.get("pen_width", 2)
        bev_size = _3dbox_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = _3dbox_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        fill_box = _3dbox_bev_settings.get("fill_box", False)
        color_table = _3dbox_bev_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        native_color_table = {
            MotionDataset.box_type_dict[k]: v for k, v in color_table.items()
        }

        corner_templates = _3dbox_bev_settings.get(
            "corner_templates",
            MotionDataset.default_bev_3dbox_corner_template)
        edge_indices = _3dbox_bev_settings.get(
            "edge_indices", MotionDataset.default_bev_3dbox_edge_indices)

        # get the transform from the referenced ego space to the BEV space
        bev_from_ego = np.array(bev_from_ego_transform)

        # draw annotations to the image
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        for i in laser_labels:
            category = i.type
            if category in native_color_table:
                pen_color = tuple(native_color_table[category])
                scale = np.diag([i.box.length, i.box.width, i.box.height, 1])
                ego_from_annotation = dwm.datasets.common.get_transform(
                    transforms3d.euler.euler2quat(
                        0, 0, i.box.heading).tolist(),
                    [i.box.center_x, i.box.center_y, i.box.center_z])
                p = bev_from_ego @ ego_from_annotation @ scale @ \
                    corner_templates_np
                if fill_box:
                    draw.polygon(
                        [(p[0, a], p[1, a]) for a, _ in edge_indices],
                        fill=pen_color, width=pen_width)
                else:
                    for a, b in edge_indices:
                        draw.line(
                            (p[0, a], p[1, a], p[0, b], p[1, b]),
                            fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def draw_polygon_bev_to_image(
        polygon: list, draw: ImageDraw, transform: np.array, pen_color: tuple,
        pen_width: int, solid: bool = True
    ):
        if len(polygon) == 0:
            return

        polygon_nodes = np.array([
            [i[0], i[1], 0, 1] for i in polygon
        ], np.float32).transpose()
        p = transform @ polygon_nodes
        draw.polygon(
            [(p[0, i], p[1, i]) for i in range(p.shape[1])],
            fill=pen_color if solid else None,
            outline=None if solid else pen_color, width=pen_width)

    @staticmethod
    def draw_line_bev_to_image(
        line: list, draw: ImageDraw, transform: np.array, pen_color: tuple,
        pen_width: int
    ):
        if len(line) == 0:
            return

        line_nodes = np.array([
            [i[0], i[1], 0, 1] for i in line
        ], np.float32).transpose()
        p = transform @ line_nodes
        for i in range(1, len(line)):
            draw.line(
                (p[0, i - 1], p[1, i - 1], p[0, i], p[1, i]),
                fill=pen_color, width=pen_width)

    @staticmethod
    def get_hdmap_bev_image(map_features, pose, hdmap_bev_settings: dict):
        pen_width = hdmap_bev_settings.get("pen_width", 2)
        bev_size = hdmap_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = hdmap_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        color_table = hdmap_bev_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the referenced ego space to the BEV space
        world_from_ego = np.array(pose.transform, np.float32).reshape(4, 4)
        bev_from_ego = np.array(bev_from_ego_transform, np.float32)
        bev_from_world = bev_from_ego @ np.linalg.inv(world_from_ego)

        # draw map elements to the image
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        type_polygons = {}
        type_polylines = {}
        for feat in map_features:
            type_ = feat.WhichOneof('feature_data')
            if (
                type_ not in color_table or
                type_ not in MotionDataset.map_element_type_dict
            ):
                continue

            type_poly = MotionDataset.map_element_type_dict[type_]
            items = getattr(getattr(feat, type_), type_poly)
            coors_3d = []
            for item in items:
                coors_3d.append([item.x, item.y, item.z])
            if len(coors_3d) > 0:
                if type_poly == "polyline":
                    if type_ not in type_polylines:
                        type_polylines[type_] = []
                    type_polylines[type_].append(coors_3d)
                else:
                    if type_ not in type_polygons:
                        type_polygons[type_] = []
                    type_polygons[type_].append(coors_3d)

        for k, v in type_polygons.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_polygon_bev_to_image(
                        i, draw, bev_from_world, c, pen_width)

        for k, v in type_polylines.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_line_bev_to_image(
                        i, draw, bev_from_world, c, pen_width)

        return image

    @staticmethod
    def get_image_description(
        image_descriptions: dict, time_list_dict: dict, scene_key: str,
        timestamp: int, camera_id: int
    ):
        nearest_time = dwm.datasets.common.find_nearest(
            time_list_dict[scene_key], timestamp, return_item=True)
        key = "{}|{}|{}".format(scene_key, nearest_time, camera_id)
        return image_descriptions[key]
    ###增加筛选平衡后的json########
    def __init__(
        self,
        fs,
        info_dict_path,
        sequence_length,
        fps_stride_tuples,
        sensor_channels=["CAM_FRONT"],
        enable_camera_transforms=False,
        enable_ego_transforms=False,
        _3dbox_image_settings=None,
        hdmap_image_settings=None,
        _3dbox_bev_settings=None,
        hdmap_bev_settings=None,
        image_description_settings=None,
        stub_key_data_dict=None,
        balanced_json_path=None,
        dataset_root=None,
        split: str = "train" 
    ):

        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.sensor_channels = sensor_channels
        self.dataset_root = dataset_root
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.stub_key_data_dict = stub_key_data_dict
        self.split =split
        self.items = []

        # ===============================
        # 1️⃣ 读取 info_dict
        # ===============================

        with open(info_dict_path, 'r') as f:
            self.sample_info_dict = json.load(f)

        # ===============================
        # 2️⃣ balanced_json（可选）
        # ===============================

        use_balance = balanced_json_path is not None

        if use_balance:
            with open(balanced_json_path, 'r') as f:
                raw_entries = json.load(f)

            print(f"[Dataset] Motion intervals loaded: {len(raw_entries)}")

            self.motion_intervals = []

            for e in raw_entries:
                self.motion_intervals.append({
                    "scene": e["seq_id"],
                    "start_idx": e["start_idx"],
                    "end_idx": e["end_idx"],
                    "start_ts": e["start_timestamp"],
                    "end_ts": e["end_timestamp"],
                    "angle": e["angle"],
                    "dist": e["dist"]
                })

            # scene -> intervals
            self.motion_intervals_by_scene = {}

            for interval in self.motion_intervals:
                scene = interval["scene"]
                if scene not in self.motion_intervals_by_scene:
                    self.motion_intervals_by_scene[scene] = []
                self.motion_intervals_by_scene[scene].append(interval)

            print("[Dataset] Interval scenes:", len(self.motion_intervals_by_scene))

        else:
            print("[Dataset] No balanced_json → no filtering")
            self.motion_intervals_by_scene = {}

        # ===============================
        # 3️⃣ enumerate windows
        # ===============================

        total_windows = 0
        matched_windows = 0

        OVERLAP_RATIO = 0.8

        for scene_id, sample_list in self.sample_info_dict.items():

            # 👉 只有用 balance 才过滤 scene
            if use_balance and scene_id not in self.motion_intervals_by_scene:
                continue

            # 👉 强烈建议排序（防止 timestamp 乱）
            sample_list = sorted(sample_list, key=lambda x: x[0])

            scene_intervals = self.motion_intervals_by_scene.get(scene_id, [])

            timestamps = [i[0] for i in sample_list]

            for fps, stride_sec in self.fps_stride_tuples:

                stride = int(stride_sec * fps)
                if stride <= 0:
                    stride = 1

                for start_idx in range(
                    0,
                    len(sample_list) - self.sequence_length + 1,
                    stride
                ):

                    end_idx = start_idx + self.sequence_length

                    window_ts_start = timestamps[start_idx]
                    window_ts_end = timestamps[end_idx - 1]

                    window_duration = window_ts_end - window_ts_start

                    total_windows += 1

                    matched_interval = None

                    # 👉 只有 use_balance 才匹配
                    if use_balance:
                        for interval in scene_intervals:

                            overlap_start = max(window_ts_start, interval["start_ts"])
                            overlap_end = min(window_ts_end, interval["end_ts"])

                            overlap = overlap_end - overlap_start

                            if overlap <= 0:
                                continue

                            if overlap >= OVERLAP_RATIO * window_duration:
                                matched_interval = interval
                                break

                        if matched_interval is None:
                            continue

                    matched_windows += 1

                    self.items.append({
                        "scene": scene_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "fps": fps,
                        "split": split,   

                        # 👉 核心兼容
                        "angle": matched_interval["angle"] if use_balance else 0.0,
                        "dist": matched_interval["dist"] if use_balance else 0.0
                    })

        # ===============================
        # 4️⃣ stats
        # ===============================

        print("[Dataset] Window enumeration finished")
        print("Total windows:", total_windows)

        if use_balance:
            print("Matched windows:", matched_windows)
        else:
            print("No filtering → using all windows")

        print("Final dataset size:", len(self.items))
        if len(self.items) > 0:
            print("[Dataset DEBUG] Example item:")
            print(self.items[0])

        # 👉 最终封装（只保留这个）
        self.items = dwm.common.SerializedReadonlyList(self.items)
            


    def __len__(self):
        return len(self.items)
    def __getitem__(self, index: int):
        item = self.items[index]
        scene_id = item["scene"]

        # 1. 检查属性是否存在
        if not hasattr(self, 'sample_info_dict'):
            raise AttributeError("sample_info_dict not initialized. Check __init__ logic.")

        # 2. 获取该场景的所有帧
        all_frames = self.sample_info_dict[scene_id]

        # 3. 截取当前窗口的帧 (根据 start_idx 和 end_idx)
        # 注意：Python 切片是左闭右开
        segment = all_frames[item["start_idx"] : item["end_idx"]]

        if len(segment) == 0:
            raise ValueError(f"No frames found for {scene_id} at {item['start_idx']}:{item['end_idx']}")

        # 4. 计算 PTS (时间戳在索引 0)
        # Waymo 时间戳是微秒，/ 1e6 换算成秒
        result = {
            "fps": torch.tensor(item["fps"]).float(),
            "angle": torch.tensor(item["angle"]).float(),
            "dist": torch.tensor(item["dist"]).float(),
           
        }
       
       # 5. 读取 TFRecord
        # 构造文件名
        scene_filename = f"segment-{scene_id}_with_camera_labels.tfrecord"
        split = item["split"]

                
        # 拼接完整路径：root / individual_files / training / filename
        if self.dataset_root:
            scene_path = os.path.join(
                self.dataset_root, 
                "individual_files", 
                    split,   # ✅ 不再写死
                scene_filename
            )
        else:
            # 如果没有 root，则尝试相对路径
            scene_path = os.path.join("individual_files", "training", scene_filename)

        # 调试打印（可选，运行成功后可删除）
        # print(f"DEBUG: Opening tfrecord from: {scene_path}")

        frames = [waymo_pb.Frame() for _ in segment]

        # 检查文件是否存在以防闪退
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Waymo record not found at: {scene_path}")

        with self.fs.open(scene_path, "rb") as f:
            for i_id, frame_info in enumerate(segment):
                # 解包你 info.json 中的 [timestamp, length, offset]
                _, length, offset = frame_info
                
                # 定位并读取这一帧的字节流
                f.seek(offset)
                frames[i_id].ParseFromString(f.read(length))

        
        # 6. 提取图像和矩阵
        images = []
        intrinsics = []
        extrinsics = []
        
        # 定义相机名字列表，方便调试时对应
        camera_names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

        for i_id in range(len(segment)):
            f_data = frames[i_id]
            
            frame_images = []
            frame_intr = []
            frame_extr = []

            for cam_name in self.sensor_channels:
                cam_id = MotionDataset.sensor_name_id_dict[cam_name]
                
                # --- A. 找图 (容错) ---
                img_data = MotionDataset.find_by_name(f_data.images, cam_id)
                if img_data:
                    with io.BytesIO(img_data.image) as f:
                        img = Image.open(f)
                        img.load()
                        frame_images.append(img)
                else:
                    # 找不到图，用黑色块占位，保证 tensor 维度还是 5
                    frame_images.append(Image.new('RGB', (448, 256), (0, 0, 0)))

                # --- B. 找矩阵 ---
                # 即使没图，通常 context 里的标定信息（矩阵）还是在的
                calib = next((c for c in f_data.context.camera_calibrations if c.name == cam_id), None)
                if calib:
                    ext = np.array(calib.extrinsic.transform).reshape(4, 4)
                    ins = np.eye(3)
                    ins[0,0], ins[1,1], ins[0,2], ins[1,2] = calib.intrinsic[0:4]
                    frame_intr.append(torch.from_numpy(ins).float())
                    frame_extr.append(torch.from_numpy(ext).float())
                else:
                    frame_intr.append(torch.eye(3))
                    frame_extr.append(torch.eye(4))

            images.append(frame_images)
            intrinsics.append(torch.stack(frame_intr))
            extrinsics.append(torch.stack(frame_extr))

        # 写入结果
        result["images"] = images
        result["camera_intrinsics"] = torch.stack(intrinsics) # [T, 5, 3, 3]
        result["camera_extrinsics"] = torch.stack(extrinsics) # [T, 5, 4, 4]
        result["camera_names"] = self.sensor_channels # 调试用
        
        scene_frame = waymo_pb.Frame()

        if self.enable_camera_transforms:
                if "images" in result:
                    camera_calibrations = [
                        [
                            MotionDataset.find_by_name(
                                i.context.camera_calibrations,
                                MotionDataset.sensor_name_id_dict[j])
                            for j in self.sensor_channels
                            if j.startswith("CAM")
                        ]
                        for i in frames
                    ]

                    ec_inv = torch.linalg.inv(
                        torch.tensor(
                            MotionDataset.extrinsic_correction,
                            dtype=torch.float32))
                    result["camera_transforms"] = torch.stack([
                        torch.stack([
                            torch.tensor(
                                j.extrinsic.transform,
                                dtype=torch.float32).reshape(4, 4) @ ec_inv
                            for j in i
                        ])
                        for i in camera_calibrations
                    ])
                    result["camera_intrinsics"] = torch.stack([
                        torch.stack([
                            dwm.datasets.common.make_intrinsic_matrix(
                                j.intrinsic[0:2], j.intrinsic[2:4], "pt")
                            for j in i
                        ])
                        for i in camera_calibrations
                    ])
                    result["image_size"] = torch.stack([
                        torch.stack([
                            torch.tensor([j.width, j.height], dtype=torch.long)
                            for j in i
                        ])
                        for i in camera_calibrations
                    ])

                if "lidar_points" in result:
                    result["lidar_transforms"] = torch.stack([
                        torch.stack([
                            torch.eye(4)
                            for j in self.sensor_channels
                            if j.startswith("LIDAR")
                        ])
                        for _ in frames
                    ])

                if self.enable_ego_transforms:
                    result["ego_transforms"] = torch.stack([
                        torch.stack([
                            torch.tensor(
                                i.pose.transform, dtype=torch.float32).reshape(4, 4)
                            for _ in self.sensor_channels
                        ])
                        for i in frames
                    ])

                if self._3dbox_image_settings is not None:
                    result["3dbox_images"] = [
                        [
                            MotionDataset.get_3dbox_image(
                                i.laser_labels,
                                MotionDataset.find_by_name(
                                    i.context.camera_calibrations,
                                    MotionDataset.sensor_name_id_dict[j]),
                                self._3dbox_image_settings)
                            for j in self.sensor_channels
                            if j.startswith("CAM")
                        ]
                        for i in frames
                    ]

                if self.hdmap_image_settings is not None:
                    result["hdmap_images"] = [
                        [
                            MotionDataset.get_hdmap_image(
                                scene_frame.map_features,
                                MotionDataset.find_by_name(
                                    i.context.camera_calibrations,
                                    MotionDataset.sensor_name_id_dict[j]),
                                i.pose, self.hdmap_image_settings)
                            for j in self.sensor_channels
                            if j.startswith("CAM")
                        ]
                        for i in frames
                    ]

                if self._3dbox_bev_settings is not None:
                    result["3dbox_bev_images"] = [
                        MotionDataset.get_3dbox_bev_image(
                            i.laser_labels, self._3dbox_bev_settings)
                        for i in frames
                        for j in self.sensor_channels
                        if j.startswith("LIDAR")
                    ]

                if self.hdmap_bev_settings is not None:
                    result["hdmap_bev_images"] = [
                        MotionDataset.get_hdmap_bev_image(
                            scene_frame.map_features, i.pose, self.hdmap_bev_settings)
                        for i in frames
                        for j in self.sensor_channels
                        if j.startswith("LIDAR")
                    ]

                if self.image_description_settings is not None:
                    image_captions = [
                        dwm.datasets.common.align_image_description_crossview([
                            MotionDataset.get_image_description(
                                self.image_descriptions, self.time_list_dict,
                                item["scene"], i[3],
                                MotionDataset.sensor_name_id_dict[j])
                            for j in self.sensor_channels
                            if "LIDAR" not in j
                        ], self.image_description_settings)
                        for i in segment
                    ]
                    result["image_description"] = [
                        [
                            dwm.datasets.common.make_image_description_string(
                                j, self.image_description_settings, self.image_desc_rs)
                            for j in i
                        ]
                        for i in image_captions
                    ]

                dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)

        # 7. 写入元数据
        if "angle" in item:
            result["angle"] = torch.tensor(item["angle"]).float()
        if "dist" in item:
            result["dist"] = torch.tensor(item["dist"]).float()
        result["scene_name"] = scene_id

        return result
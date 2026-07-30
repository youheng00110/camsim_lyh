import copy
import json
from pathlib import Path

config_dir = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim"
)

new_base_dataset = {
    "_class_name": "torch.utils.data.ConcatDataset",
    "datasets": [
        {
            "_class_name": "dwm.datasets.argoverse.MotionDataset",
            "fs": {
                "_class_name": "dwm.fs.dirfs.DirFileSystem",
                "fs": {
                    "_class_name": "dwm.fs.dirfs.DirFileSystem",
                    "path": (
                        "/inspire/qb-ilm/project/advanced-machine-learning/"
                        "yanjunchi-24040/camsim_lyh/avrgo2_link"
                    )
                },
                "enable_cached_info": True
            },
            "split": "val",
            "dataset_root": (
                "/inspire/qb-ilm/project/advanced-machine-learning/"
                "yanjunchi-24040/camsim_lyh/avrgo2_link"
            ),
            "index_json_path": (
                "/inspire/qb-ilm/project/advanced-machine-learning/"
                "yanjunchi-24040/camsim_lyh/avrgo2_json"
            ),
            "sequence_length": 19,
            "fps_stride_tuples": [
                [
                    6,
                    2
                ]
            ],
            "sensor_channels": [
                "lidar",
                "cameras/ring_side_left",
                "cameras/ring_front_left",
                "cameras/ring_front_right",
                "cameras/ring_side_right"
            ],
            "enable_camera_transforms": True,
            "enable_ego_transforms": True,
            "_3dbox_image_settings": {},
            "hdmap_image_settings": {},
            "image_description_settings": {
                "path": (
                    "/inspire/qb-ilm/project/advanced-machine-learning/"
                    "yanjunchi-24040/camsim_lyh/"
                    "av2_sensor_caption_v2/"
                    "av2_sensor_caption_v2_val.json"
                ),
                "time_list_dict_path": (
                    "/inspire/qb-ilm/project/advanced-machine-learning/"
                    "yanjunchi-24040/camsim_lyh/"
                    "av2_sensor_caption_v2/"
                    "av2_sensor_caption_v2_times_val.json"
                ),
                "align_keys": [
                    "time",
                    "weather"
                ],
                "reorder_keys": True,
                "drop_rates": {
                    "environment": 0.04,
                    "objects": 0.08,
                    "image_description": 0.16
                }
            },
            "stub_key_data_dict": {
                "crossview_mask": [
                    "content",
                    {
                        "_class_name": "torch.tensor",
                        "data": {
                            "_class_name": "json.loads",
                            "s": (
                                "[[1,1,0,0],"
                                "[1,1,1,0],"
                                "[0,1,1,1],"
                                "[0,0,1,1]]"
                            )
                        },
                        "dtype": {
                            "_class_name": "get_class",
                            "class_name": "torch.bool"
                        }
                    }
                ],
                "dataset_tag": [
                    "content",
                    {
                        "_class_name": "torch.tensor",
                        "data": 3,
                        "dtype": {
                            "_class_name": "get_class",
                            "class_name": "torch.int64"
                        }
                    }
                ]
            }
        }
    ]
}

source_files = sorted(config_dir.glob("*preview.json"))

if not source_files:
    raise RuntimeError(f"没有找到 *preview.json：{config_dir}")

success_count = 0

for source_path in source_files:
    with source_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    validation_dataset = data.get("validation_dataset")

    if not isinstance(validation_dataset, dict):
        print(
            f"[跳过] {source_path.name}: "
            "validation_dataset 不存在或不是字典"
        )
        continue

    validation_dataset["base_dataset"] = copy.deepcopy(new_base_dataset)

    output_path = source_path.with_name(
        f"{source_path.stem}4cam.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )
        file.write("\n")

    success_count += 1
    print(f"[生成] {source_path.name} -> {output_path.name}")

print()
print(f"共找到原始配置：{len(source_files)} 个")
print(f"成功生成新配置：{success_count} 个")
print("原始 preview 配置未修改。")

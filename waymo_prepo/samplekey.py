from dwm.datasets.waymo import WaymoDataset

dataset = WaymoDataset(
    root=WAYMO_ROOT,
    info_path="prepro_waymo/training.info.json",
    split="training",
    camera_only=True
)

sample = dataset[0]
print(sample.keys())
sample = dataset[0]
print(sample.keys())
print("frame_idx:", sample["frame_idx"])
for cam, img in sample["images"].items():
    print(cam, img.shape)

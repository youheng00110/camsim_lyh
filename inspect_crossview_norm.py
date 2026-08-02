#!/usr/bin/env python3
import csv
from pathlib import Path

import torch

ckpt_a_path = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/train_nuplantokenearly/"
    "checkpoints/24000.pth"
)
ckpt_b_path = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/train_nuplantokenearly/"
    "checkpoints/30000.pth"
)
output_path = Path("crossview_norm_detailed_24000_vs_30000.csv")

state_a = torch.load(
    ckpt_a_path,
    map_location="cpu",
    weights_only=False,
)
state_b = torch.load(
    ckpt_b_path,
    map_location="cpu",
    weights_only=False,
)

# 你的 checkpoint 参数就在根目录。
normalized_a = {}
for original_key, value in state_a.items():
    if not isinstance(value, torch.Tensor):
        continue

    key = original_key
    while key.startswith("module.") or key.startswith("_orig_mod."):
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]

    normalized_a[key] = value

normalized_b = {}
for original_key, value in state_b.items():
    if not isinstance(value, torch.Tensor):
        continue

    key = original_key
    while key.startswith("module.") or key.startswith("_orig_mod."):
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]

    normalized_b[key] = value

common_keys = sorted(
    key
    for key in normalized_a
    if key in normalized_b
    and "crossview" in key.lower()
    and "norm" in key.lower()
)

rows = []

for key in common_keys:
    tensor_a = normalized_a[key].detach().float().reshape(-1)
    tensor_b = normalized_b[key].detach().float().reshape(-1)

    if tensor_a.shape != tensor_b.shape:
        print(
            f"跳过形状不一致参数：{key}，"
            f"A={tuple(tensor_a.shape)}，B={tuple(tensor_b.shape)}"
        )
        continue

    difference = tensor_b - tensor_a
    absolute_difference = difference.abs()

    a_l2 = tensor_a.norm(p=2).item()
    b_l2 = tensor_b.norm(p=2).item()
    difference_l2 = difference.norm(p=2).item()
    relative_l2 = difference_l2 / max(a_l2, 1e-12)

    denominator = tensor_a.norm(p=2) * tensor_b.norm(p=2)
    if denominator.item() > 1e-12:
        cosine = (
            torch.dot(tensor_a, tensor_b) / denominator
        ).item()
    else:
        cosine = float("nan")

    rows.append(
        {
            "key": key,
            "type": "weight" if key.endswith(".weight") else "bias",
            "numel": tensor_a.numel(),

            "a_mean_24000": tensor_a.mean().item(),
            "b_mean_30000": tensor_b.mean().item(),
            "mean_shift": difference.mean().item(),

            "a_abs_mean_24000": tensor_a.abs().mean().item(),
            "b_abs_mean_30000": tensor_b.abs().mean().item(),
            "mean_absolute_change": absolute_difference.mean().item(),

            "a_std_24000": tensor_a.std(unbiased=False).item(),
            "b_std_30000": tensor_b.std(unbiased=False).item(),

            "a_min_24000": tensor_a.min().item(),
            "a_max_24000": tensor_a.max().item(),
            "b_min_30000": tensor_b.min().item(),
            "b_max_30000": tensor_b.max().item(),

            "a_l2_norm_24000": a_l2,
            "b_l2_norm_30000": b_l2,
            "l2_difference": difference_l2,
            "relative_l2_difference": relative_l2,

            "max_absolute_change": absolute_difference.max().item(),
            "cosine_similarity": cosine,
        }
    )

rows.sort(
    key=lambda row: row["relative_l2_difference"],
    reverse=True,
)

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 150)
print("24000 初值与 24000→30000 变化")
print("=" * 150)
print(
    f"{'类型':<7}"
    f"{'24000均值':>13}"
    f"{'30000均值':>13}"
    f"{'均值变化':>13}"
    f"{'24000绝对均值':>16}"
    f"{'平均绝对变化':>16}"
    f"{'24000 L2':>13}"
    f"{'L2变化':>13}"
    f"{'相对变化':>11}  "
    f"参数名"
)

for row in rows:
    print(
        f"{row['type']:<7}"
        f"{row['a_mean_24000']:13.5e}"
        f"{row['b_mean_30000']:13.5e}"
        f"{row['mean_shift']:13.5e}"
        f"{row['a_abs_mean_24000']:16.5e}"
        f"{row['mean_absolute_change']:16.5e}"
        f"{row['a_l2_norm_24000']:13.5e}"
        f"{row['l2_difference']:13.5e}"
        f"{row['relative_l2_difference']:10.2%}  "
        f"{row['key']}"
    )

print()
print(f"参数数量：{len(rows)}")
print(f"详细结果：{output_path.resolve()}")

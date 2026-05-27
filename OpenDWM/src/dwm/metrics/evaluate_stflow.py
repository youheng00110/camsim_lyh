import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dwm.metrics.stflow import STFlowEvaluator


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate ST-Flow consistency from generated multi-view manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to stflow_manifest.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save result JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--loftr-confidence",
        type=float,
        default=0.2,
    )
    return parser


def load_manifest_items(manifest_path, max_videos):
    items = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue

            items.append(json.loads(line))

            if max_videos is not None and len(items) >= max_videos:
                break

    return items


def aggregate_video_results(video_results):
    scalar_keys = [
        "temporal_l1",
        "cross_epi_px",
        "cycle_epi_px",
        "stflow_error",
        "stflow_score",
        "num_temporal_edges",
        "num_cross_edges",
        "num_cycle_edges",
    ]

    output = {
        "num_videos": len(video_results),
        "videos": video_results,
        "mean": {},
    }

    for key in scalar_keys:
        values = []
        for result in video_results:
            value = result.get(key, float("nan"))
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            values.append(float(value))

        output["mean"][key] = float(np.mean(values)) if len(values) > 0 else float("nan")

    pair_values = {}
    for result in video_results:
        for pair_key, pair_result in result.get("pair_stats", {}).items():
            if pair_key not in pair_values:
                pair_values[pair_key] = {
                    "cross_epi_px": [],
                    "cycle_epi_px": [],
                    "match_count": [],
                }

            for metric_key in pair_values[pair_key]:
                metric_value = pair_result.get(metric_key, float("nan"))
                if metric_value is None:
                    continue
                if isinstance(metric_value, float) and np.isnan(metric_value):
                    continue
                pair_values[pair_key][metric_key].append(float(metric_value))

    output["camera_pair_mean"] = {}
    for pair_key, metric_values in pair_values.items():
        output["camera_pair_mean"][pair_key] = {}
        for metric_key, values in metric_values.items():
            output["camera_pair_mean"][pair_key][metric_key] = (
                float(np.mean(values)) if len(values) > 0 else float("nan")
            )

    return output


def main():
    parser = create_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest_dir = str(manifest_path.parent)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = STFlowEvaluator(
        device=args.device,
        frame_stride=args.frame_stride,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        loftr_confidence=args.loftr_confidence,
    )

    manifest_items = load_manifest_items(args.manifest, args.max_videos)
    print(f"[stflow] Loaded {len(manifest_items)} videos from {args.manifest}")

    video_results = []
    for index, item in enumerate(manifest_items):
        with torch.no_grad():
            result = evaluator.evaluate_video(item, manifest_dir)

        video_results.append(result)
        print(
            "[stflow] {}/{} {} | score={:.3f} temp={:.5f} cross={:.3f} cycle={:.3f}".format(
                index + 1,
                len(manifest_items),
                result.get("video_id", ""),
                result["stflow_score"],
                result["temporal_l1"],
                result["cross_epi_px"],
                result["cycle_epi_px"],
            )
        )

    output = aggregate_video_results(video_results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[stflow] Saved result to {output_path}")
    print(json.dumps(output["mean"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
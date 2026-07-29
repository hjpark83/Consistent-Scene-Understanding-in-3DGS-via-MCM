#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def to_bool(value: str) -> bool:
    return value.lower() not in {"0", "false", "no", "off"}


ENV_OVERRIDES: tuple[tuple[str, str, Callable[[str], object]], ...] = (
    ("MCM_MIN_MASK_AREA", "min_mask_area", int),
    ("MCM_LOG_SIGMA", "log_sigma", float),
    ("MCM_LAPLACIAN_KSIZE", "laplacian_ksize", int),
    ("MCM_FEATURE_WEIGHT", "feature_weight", float),
    ("MCM_FEATURE_SIM_THRESHOLD", "feature_sim_threshold", float),
    ("MCM_EDGE_THRESHOLD", "edge_strength_threshold", float),
    ("MCM_EDGE_PENALTY", "edge_penalty", float),
    ("MCM_MERGE_SCORE_THRESHOLD", "merge_score_threshold", float),
    ("MCM_NORMALIZE_MERGE_SCORE", "normalize_merge_score", to_bool),
    ("MCM_BOUNDARY_REDUCTION", "boundary_reduction", str),
    ("MCM_BOUNDARY_PERCENTILE", "boundary_percentile", float),
    ("MCM_ADJACENCY_DILATION", "adjacency_dilation", int),
    ("MCM_MIN_CONTACT_RATIO", "min_contact_ratio", float),
    ("MCM_MAX_MERGE_ITERATIONS", "max_merge_iterations", int),
    ("MCM_FEATURE_MARGIN", "feature_margin", float),
    ("MCM_ADJACENCY_MAX_BBOX_GAP", "adjacency_max_bbox_gap", int),
    ("MCM_ENABLE_CONTAINMENT_MERGE", "enable_containment_merge", to_bool),
    ("MCM_CONTAINMENT_RATIO", "containment_ratio_threshold", float),
    ("MCM_CONTAINMENT_FEATURE", "containment_feature_threshold", float),
    ("MCM_USE_DEPTH", "use_depth", to_bool),
    ("MCM_DEPTH_METHOD", "depth_method", str),
    ("MCM_DEPTH_WEIGHT", "depth_weight", float),
    ("MCM_DEPTH_BOUNDARY_WEIGHT", "depth_boundary_weight", float),
    ("MCM_DEPTH_DIFF_THRESHOLD", "depth_diff_threshold", float),
    ("MCM_DEPTH_AFFINITY_SIGMA2", "depth_affinity_sigma2", float),
    ("MCM_DINO_MAX_LONG_EDGE", "dino_max_long_edge", int),
    ("MCM_DINO_TILE_SIZE", "dino_tile_size", int),
    ("MCM_DINO_TILE_STRIDE", "dino_tile_stride", int),
    ("MCM_SAM_MAX_LONG_EDGE", "sam_max_long_edge", int),
    ("MCM_DEPTH_GRADIENT_SIGMA", "depth_gradient_sigma", float),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM + LoG + DINO feature field segmentation for static scenes."
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing RGB images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for mask npz files.")
    parser.add_argument("--sam-checkpoint", default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint.")
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda", help="Device for DINO feature extractor.")
    parser.add_argument("--sam-device", default=None, help="Device for SAM (defaults to --device).")
    parser.add_argument("--sam-fallback-device", default="cpu", help="Fallback device if SAM OOMs on the primary device.")
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--dino-cache-dir", type=Path, default=None, help="Directory to cache per-image DINO features.")
    parser.add_argument("--depth-cache-dir", type=Path, default=None, help="Directory with precomputed depth maps (.npy files).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images whose outputs already exist.")
    parser.add_argument("--visualize", action="store_true", help="Save visualization overlays for each frame.")
    parser.add_argument("--visualize-dir", type=Path, default=None, help="Directory for visualization outputs (defaults to output-dir/viz).")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Image extensions to process.",
    )
    return parser.parse_args()


def collect_images(image_dir: Path, extensions: Sequence[str]) -> List[Path]:
    exts = {ext.lower() for ext in extensions}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def apply_env_overrides(config) -> None:
    for env_name, attr_name, caster in ENV_OVERRIDES:
        if env_name in os.environ:
            setattr(config, attr_name, caster(os.environ[env_name]))


def build_feature_field_config(args, cache_dir: Optional[Path], config_cls):
    config = config_cls(
        sam_checkpoint=str(args.sam_checkpoint),
        sam_model_type=args.sam_model_type,
        device=args.device,
        sam_device=args.sam_device,
        sam_fallback_device=args.sam_fallback_device,
        dino_model_name=args.dino_model_name,
        dino_fallback_models=["dinov2_vits14_reg"],
        dino_cache_dir=str(cache_dir) if cache_dir else None,
        depth_cache_dir=str(args.depth_cache_dir) if args.depth_cache_dir else None,
        depth_gradient_sigma=1.5,
    )
    apply_env_overrides(config)
    return config


def save_result(output_path: Path, image_name: str, result: dict) -> None:
    regions = result["regions"]
    edge_map = result["edge_map"]
    depth_map = result.get("depth_map", None)
    initial_masks = result.get("initial_masks", None)

    if not regions:
        save_dict = {
            "masks": np.zeros((0, 1, 1), dtype=bool),
            "features": np.zeros((0, 1), dtype=np.float32),
            "areas": np.zeros((0,), dtype=np.int32),
            "bboxes": np.zeros((0, 4), dtype=np.int32),
            "source_ids": np.zeros((0, 1), dtype=np.int32),
            "edge_map": edge_map.astype(np.float32),
            "image_name": image_name,
        }
        if initial_masks is not None:
            save_dict["initial_masks"] = np.asarray(initial_masks, dtype=bool)
        if depth_map is not None:
            save_dict["depth_map"] = depth_map.astype(np.float32)
        np.savez_compressed(output_path, **save_dict)
        return

    masks = np.stack([region.mask for region in regions], axis=0).astype(bool)
    features = torch.stack([region.feature for region in regions]).cpu().numpy().astype(np.float32)
    areas = np.array([region.area for region in regions], dtype=np.int32)
    bboxes = np.array([region.bbox for region in regions], dtype=np.int32)
    max_sources = max(len(region.source_ids) for region in regions)
    source_ids = -np.ones((len(regions), max_sources), dtype=np.int32)
    for idx, region in enumerate(regions):
        count = len(region.source_ids)
        source_ids[idx, :count] = region.source_ids

    save_dict = {
        "masks": masks,
        "features": features,
        "areas": areas,
        "bboxes": bboxes,
        "source_ids": source_ids,
        "edge_map": edge_map.astype(np.float32),
        "image_name": image_name,
    }
    if initial_masks is not None:
        save_dict["initial_masks"] = np.asarray(initial_masks, dtype=bool)
    if depth_map is not None:
        save_dict["depth_map"] = depth_map.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)


def main() -> None:
    args = parse_args()
    from segmentation import MaskFeatureConfig, MaskFeaturePipeline, save_feature_field_visualizations

    image_dir: Path = args.image_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = None
    if args.visualize:
        viz_dir = args.visualize_dir or (output_dir / "visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
    cache_dir: Optional[Path] = args.dino_cache_dir
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir, args.extensions)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir} with extensions {args.extensions}")

    config = build_feature_field_config(args, cache_dir, MaskFeatureConfig)
    pipeline = MaskFeaturePipeline(config)

    for image_path in tqdm(images, desc="Feature-field segmentation"):
        output_path = output_dir / f"{image_path.stem}.npz"
        if args.skip_existing and output_path.exists():
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        result = pipeline.process_image(image, image_name=image_path.stem)
        save_result(output_path, image_path.stem, result)
        if viz_dir is not None:
            save_feature_field_visualizations(
                image=image,
                regions=result["regions"],
                edge_map=result["edge_map"],
                out_dir=viz_dir,
                image_name=image_path.stem,
            )

    print(f"\n Saved refined masks to {output_dir}")


if __name__ == "__main__":
    main()

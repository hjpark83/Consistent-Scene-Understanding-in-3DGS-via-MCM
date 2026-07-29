#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
import sys
import json

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, get_combined_args
from scene import GaussianModel, Scene
from segmentation import load_feature_field_directory
from utils.feature_lifting import lift_feature_field_masks_to_gaussians


def smooth_features_by_mask_id(features: torch.Tensor, mask_ids: torch.Tensor, blend: float = 1.0) -> torch.Tensor:
    assigned = mask_ids >= 0
    if not assigned.any():
        return features

    blend = max(0.0, min(1.0, float(blend)))
    smoothed = features.detach().clone()
    unique_ids = torch.unique(mask_ids[assigned])

    for obj_id in unique_ids:
        obj_mask = mask_ids == obj_id
        if obj_mask.sum() == 0:
            continue
        prototype = features[obj_mask].mean(dim=0, keepdim=True)
        smoothed[obj_mask] = blend * prototype + (1.0 - blend) * features[obj_mask]

    return smoothed


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    parser = ArgumentParser(
        description="Assign MCM mask IDs to an already-trained Gaussian point cloud."
    )
    ModelParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument(
        "--output_ply",
        default="",
        type=str,
        help="Optional output PLY path. Defaults to point_cloud.with_mask_ids.ply unless --overwrite_point_cloud is set.",
    )
    parser.add_argument(
        "--overwrite_point_cloud",
        action="store_true",
        help="Overwrite the iteration point_cloud.ply after creating a .bak copy.",
    )
    parser.add_argument(
        "--matching_max_view_gap",
        default=None,
        type=int,
        help="Only match masks whose training-view indices differ by at most this value.",
    )
    parser.add_argument(
        "--matching_topk_per_mask",
        default=None,
        type=int,
        help="Keep at most this many strongest cross-view edges per mask.",
    )
    parser.add_argument(
        "--disable_matching_one_to_one",
        action="store_true",
        help="Disable greedy one-to-one matching per view pair.",
    )
    parser.add_argument(
        "--disable_geometry_matching",
        action="store_true",
        help="Disable 3D co-visibility overlap in cross-view matching.",
    )
    parser.add_argument(
        "--geometry_sample_points",
        default=None,
        type=int,
        help="Number of Gaussians sampled for 3D co-visibility matching.",
    )
    parser.add_argument(
        "--geometry_max_points_per_mask",
        default=None,
        type=int,
        help="Maximum sampled Gaussian IDs stored per 2D mask for matching.",
    )
    parser.add_argument(
        "--disable_zbuffer_visibility",
        action="store_true",
        help="Disable visibility gating when projecting Gaussians into 2D masks.",
    )
    parser.add_argument(
        "--zbuffer_abs_tolerance",
        default=0.02,
        type=float,
        help="Absolute depth tolerance for z-buffer visibility gating.",
    )
    parser.add_argument(
        "--zbuffer_rel_tolerance",
        default=0.03,
        type=float,
        help="Relative depth tolerance for z-buffer visibility gating.",
    )
    parser.add_argument(
        "--no_smooth_features_by_mask",
        action="store_true",
        help="Keep raw lifted DINO features instead of replacing each global object with its prototype feature.",
    )
    parser.add_argument(
        "--feature_smoothing_blend",
        default=1.0,
        type=float,
        help="Blend between raw lifted features and global mask prototype features. 1.0 is fully object-wise.",
    )
    parser.add_argument(
        "--mask_vote_area_alpha",
        default=0.5,
        type=float,
        help="Area-aware 3D mask voting exponent. Higher values favor compact object masks over broad background masks.",
    )
    parser.add_argument(
        "--mask_vote_max_weight",
        default=4.0,
        type=float,
        help="Maximum up/down weight applied by area-aware mask voting.",
    )
    parser.add_argument(
        "--mask_vote_min_count",
        default=1,
        type=int,
        help="Minimum raw per-Gaussian votes required for the winning global mask ID.",
    )
    parser.add_argument(
        "--mask_vote_min_score_ratio",
        default=0.0,
        type=float,
        help="Minimum fraction of weighted votes held by the winning global mask ID.",
    )
    parser.add_argument(
        "--mask_vote_min_margin",
        default=0.0,
        type=float,
        help="Minimum normalized score margin between the best and second-best global mask IDs.",
    )
    args = get_combined_args(parser)

    if not args.feature_field_dir:
        raise ValueError("--feature_field_dir is required.")

    feature_field_dir = Path(args.feature_field_dir)
    if not feature_field_dir.is_dir():
        raise FileNotFoundError(f"Feature-field directory not found: {feature_field_dir}")

    matching_max_view_gap_arg = getattr(args, "matching_max_view_gap", None)
    matching_topk_per_mask_arg = getattr(args, "matching_topk_per_mask", None)
    geometry_sample_points_arg = getattr(args, "geometry_sample_points", None)
    geometry_max_points_per_mask_arg = getattr(args, "geometry_max_points_per_mask", None)

    matching_max_view_gap = (
        int(matching_max_view_gap_arg)
        if matching_max_view_gap_arg is not None
        else int(getattr(args, "feature_field_matching_max_view_gap", 10))
    )
    matching_topk_per_mask = (
        int(matching_topk_per_mask_arg)
        if matching_topk_per_mask_arg is not None
        else int(getattr(args, "feature_field_matching_topk_per_mask", 3))
    )
    matching_one_to_one = (
        False
        if args.disable_matching_one_to_one
        else bool(getattr(args, "feature_field_matching_one_to_one", True))
    )
    use_geometry_matching = (
        False
        if args.disable_geometry_matching
        else bool(getattr(args, "feature_field_use_geometry_matching", True))
    )
    geometry_sample_points = (
        int(geometry_sample_points_arg)
        if geometry_sample_points_arg is not None
        else int(getattr(args, "feature_field_matching_geometry_sample_points", 120000))
    )
    geometry_max_points_per_mask = (
        int(geometry_max_points_per_mask_arg)
        if geometry_max_points_per_mask_arg is not None
        else int(getattr(args, "feature_field_matching_geometry_max_points_per_mask", 2048))
    )
    geometry_weight = float(getattr(args, "feature_field_matching_geometry_weight", 0.35))
    geometry_threshold = float(getattr(args, "feature_field_matching_geometry_threshold", 0.05))
    geometry_semantic_floor = float(getattr(args, "feature_field_matching_geometry_semantic_floor", 0.60))
    prevent_view_conflicts = bool(getattr(args, "feature_field_prevent_view_conflicts", True))
    use_zbuffer_visibility = not bool(getattr(args, "disable_zbuffer_visibility", False))

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()

    feature_field_views = load_feature_field_directory(feature_field_dir)
    if not feature_field_views:
        raise ValueError(f"No .npz feature-field files found in {feature_field_dir}")

    gaussian_features, reliability_weights, stats, gaussian_mask_ids = lift_feature_field_masks_to_gaussians(
        gaussians.get_xyz,
        train_cameras,
        feature_field_views,
        variance_threshold=float(args.feature_field_variance),
        min_views=int(args.feature_field_min_views),
        variance_alpha=float(args.feature_field_variance_alpha),
        use_improved_matching=bool(args.use_improved_matching),
        min_consensus_views=int(args.min_consensus_views),
        matching_threshold=float(args.feature_field_matching_threshold),
        same_view_matching_threshold=(
            float(args.feature_field_same_view_matching_threshold)
            if float(args.feature_field_same_view_matching_threshold) >= 0
            else None
        ),
        use_multiview_refinement=bool(args.use_multiview_refinement),
        filter_mask_ids_by_reliability=False,
        matching_max_view_gap=matching_max_view_gap,
        matching_topk_per_mask=matching_topk_per_mask,
        matching_one_to_one=matching_one_to_one,
        use_geometry_matching=use_geometry_matching,
        matching_geometry_sample_points=geometry_sample_points,
        matching_geometry_max_points_per_mask=geometry_max_points_per_mask,
        matching_geometry_weight=geometry_weight,
        matching_geometry_threshold=geometry_threshold,
        matching_geometry_semantic_floor=geometry_semantic_floor,
        prevent_view_conflicts=prevent_view_conflicts,
        use_zbuffer_visibility=use_zbuffer_visibility,
        zbuffer_abs_tolerance=float(args.zbuffer_abs_tolerance),
        zbuffer_rel_tolerance=float(args.zbuffer_rel_tolerance),
        mask_vote_area_alpha=float(args.mask_vote_area_alpha),
        mask_vote_max_weight=float(args.mask_vote_max_weight),
        mask_vote_min_count=int(args.mask_vote_min_count),
        mask_vote_min_score_ratio=float(args.mask_vote_min_score_ratio),
        mask_vote_min_margin=float(args.mask_vote_min_margin),
    )

    if not args.no_smooth_features_by_mask:
        gaussian_features = smooth_features_by_mask_id(
            gaussian_features,
            gaussian_mask_ids.to(gaussian_features.device),
            blend=args.feature_smoothing_blend,
        )

    gaussians.initialize_dino_features(
        gaussian_features,
        feature_dim=int(gaussian_features.shape[1]),
        requires_grad=False,
    )
    gaussians.set_dino_feature_targets(gaussian_features, reliability_weights)
    gaussians.mask_ids = gaussian_mask_ids.to(gaussians.get_xyz.device).long()
    assigned = int((gaussians.mask_ids >= 0).sum().item())
    unique = int(torch.unique(gaussians.mask_ids[gaussians.mask_ids >= 0]).numel())
    feature_norm = gaussians.get_dino_features.detach().norm(dim=1)
    nonzero_feature_count = int((feature_norm > 1e-8).sum().item())

    point_cloud_dir = Path(args.model_path) / "point_cloud" / f"iteration_{scene.loaded_iter}"
    source_ply = point_cloud_dir / "point_cloud.ply"
    if args.output_ply:
        output_ply = Path(args.output_ply)
    elif args.overwrite_point_cloud:
        output_ply = source_ply
    else:
        output_ply = point_cloud_dir / "point_cloud.with_mask_ids.ply"

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    if output_ply == source_ply and source_ply.exists():
        backup_ply = source_ply.with_suffix(".ply.bak")
        if not backup_ply.exists():
            shutil.copy2(source_ply, backup_ply)

    gaussians.save_ply(str(output_ply))

    mapping_path = Path(args.model_path) / f"mask_id_mapping_iter_{scene.loaded_iter}.json"
    mapping_payload = {
        "iteration": int(scene.loaded_iter),
        "feature_field_dir": str(feature_field_dir),
        "point_cloud": str(output_ply),
        "assigned_gaussians": assigned,
        "unique_mask_ids": unique,
        "dino_feature_dim": int(gaussian_features.shape[1]),
        "nonzero_dino_feature_gaussians": nonzero_feature_count,
        "features_smoothed_by_mask": not args.no_smooth_features_by_mask,
        "feature_smoothing_blend": float(args.feature_smoothing_blend),
        "view_image_names": stats.get("view_image_names", []),
        "local_to_global_map": stats.get("local_to_global_map", []),
        "global_mask_count": stats.get("global_mask_count", unique),
        "matching": {
            "threshold": float(args.feature_field_matching_threshold),
            "same_view_threshold": (
                None
                if float(args.feature_field_same_view_matching_threshold) < 0
                else float(args.feature_field_same_view_matching_threshold)
            ),
            "max_view_gap": matching_max_view_gap,
            "topk_per_mask": matching_topk_per_mask,
            "one_to_one": matching_one_to_one,
            "use_geometry": use_geometry_matching,
            "geometry_sample_points": geometry_sample_points,
            "geometry_max_points_per_mask": geometry_max_points_per_mask,
            "geometry_weight": geometry_weight,
            "geometry_threshold": geometry_threshold,
            "geometry_semantic_floor": geometry_semantic_floor,
            "prevent_view_conflicts": prevent_view_conflicts,
            "use_zbuffer_visibility": use_zbuffer_visibility,
            "zbuffer_abs_tolerance": float(args.zbuffer_abs_tolerance),
            "zbuffer_rel_tolerance": float(args.zbuffer_rel_tolerance),
            "mask_vote_area_alpha": float(args.mask_vote_area_alpha),
            "mask_vote_max_weight": float(args.mask_vote_max_weight),
            "mask_vote_min_count": int(args.mask_vote_min_count),
            "mask_vote_min_score_ratio": float(args.mask_vote_min_score_ratio),
            "mask_vote_min_margin": float(args.mask_vote_min_margin),
        },
        "voted_id_counts": stats.get("voted_id_counts", {}),
    }
    with open(mapping_path, "w") as f:
        json.dump(mapping_payload, f, indent=2)

if __name__ == "__main__":
    main()

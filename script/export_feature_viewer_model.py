#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def robust_minmax(x, low=1.0, high=99.0):
    lo = np.percentile(x, low, axis=0)
    hi = np.percentile(x, high, axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    return np.clip((x - lo) / scale, 0.0, 1.0)


def l2_normalize(x, eps=1e-8):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def fit_pca(features, max_samples=200000, seed=0):
    if features.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(features.shape[0], size=max_samples, replace=False)
        sample = features[sample_idx]
    else:
        sample = features

    mean = sample.mean(axis=0, keepdims=True)
    centered = sample - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    return mean.astype(np.float32), basis.astype(np.float32)


def pca_to_rgb(features, valid_mask, max_samples=200000, seed=0):
    valid_features = features[valid_mask]
    if valid_features.shape[0] < 3:
        raise ValueError("Need at least three valid feature vectors for PCA visualization.")

    mean, basis = fit_pca(valid_features, max_samples=max_samples, seed=seed)
    projected = (features - mean) @ basis
    rgb = np.zeros((features.shape[0], 3), dtype=np.float32)
    rgb_valid = robust_minmax(projected[valid_mask]).astype(np.float32)

    # Flip PCA channels to a stable, high-contrast orientation.
    for c in range(3):
        if rgb_valid[:, c].mean() < 0.5:
            rgb_valid[:, c] = 1.0 - rgb_valid[:, c]
    rgb[valid_mask] = rgb_valid
    rgb[~valid_mask] = np.array([0.08, 0.08, 0.08], dtype=np.float32)
    return rgb, mean.squeeze(0), basis


def apply_mask_prototype_blend(features, mask_ids, blend, min_points):
    if blend <= 0.0 or mask_ids is None:
        return features

    blended = features.copy()
    valid_ids = sorted(int(x) for x in np.unique(mask_ids) if int(x) >= 0)
    for mask_id in valid_ids:
        mask = mask_ids == mask_id
        if int(mask.sum()) < min_points:
            continue
        prototype = features[mask].mean(axis=0, keepdims=True)
        blended[mask] = (1.0 - blend) * features[mask] + blend * prototype
    return blended


def load_vertex_arrays(ply_path):
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    property_names = [p.name for p in vertex.properties]

    dino_names = sorted(
        [name for name in property_names if name.startswith("dino_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not dino_names:
        raise ValueError(f"No dino_* feature properties found in {ply_path}")

    dino = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in dino_names], axis=1)
    if not np.isfinite(dino).all():
        raise ValueError("DINO feature array contains NaN or Inf values.")

    mask_ids = None
    if "mask_id" in property_names:
        mask_ids = np.asarray(vertex["mask_id"]).astype(np.int64)

    return ply, vertex, property_names, dino, mask_ids


def write_modified_ply(ply, property_names, rgb, output_path, zero_sh_rest=True):
    vertex = ply["vertex"]
    dtype = [(name, vertex.data.dtype.fields[name][0]) for name in property_names]
    modified = np.empty(vertex.count, dtype=dtype)

    for name in property_names:
        modified[name] = np.asarray(vertex[name])

    sh = rgb_to_sh(rgb.astype(np.float32))
    for channel in range(3):
        key = f"f_dc_{channel}"
        if key not in property_names:
            raise ValueError(f"Missing required property {key}")
        modified[key] = sh[:, channel]

    if zero_sh_rest:
        for name in property_names:
            if name.startswith("f_rest_"):
                modified[name] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(modified, "vertex")], text=ply.text).write(str(output_path))


def rewrite_cfg_args(src_cfg, dst_cfg, output_model):
    text = src_cfg.read_text()
    replacement = f"model_path='{output_model}'"
    text = re.sub(r"model_path='[^']*'", replacement, text)
    dst_cfg.write_text(text)


def copy_viewer_sidecar_files(source_model, output_model):
    source_model = Path(source_model)
    output_model = Path(output_model)
    output_model.mkdir(parents=True, exist_ok=True)

    for filename in ("cameras.json", "input.ply"):
        src = source_model / filename
        if src.exists():
            shutil.copy2(src, output_model / filename)

    src_cfg = source_model / "cfg_args"
    if src_cfg.exists():
        rewrite_cfg_args(src_cfg, output_model / "cfg_args", str(output_model))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a feature-colored 3DGS model for the standard viewer."
    )
    parser.add_argument("-m", "--model_path", required=True, help="Trained MCM/3DGS model directory.")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--output_model",
        default=None,
        help="Output viewer model directory. Default: <dataset_output>/render/feature_viewer_model",
    )
    parser.add_argument(
        "--prototype_blend",
        type=float,
        default=0.45,
        help="Blend each Gaussian feature toward its global mask prototype. 0 keeps raw DINO, 1 makes each mask flat.",
    )
    parser.add_argument("--prototype_min_points", type=int, default=64)
    parser.add_argument("--max_pca_samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep_sh_rest",
        action="store_true",
        help="Keep original high-order RGB SH residuals instead of zeroing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    src_ply = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not src_ply.exists():
        raise FileNotFoundError(src_ply)

    if args.output_model is None:
        dataset_output = model_path.parent
        output_model = dataset_output / "render" / "feature_viewer_model"
    else:
        output_model = Path(args.output_model)

    ply, _, property_names, dino, mask_ids = load_vertex_arrays(src_ply)
    valid = np.linalg.norm(dino, axis=1) > 1e-8
    if int(valid.sum()) == 0:
        raise ValueError(
            "All DINO features are zero. Re-run assign_mask_ids_to_point_cloud.py "
            "so the PLY contains lifted feature vectors."
        )

    features = l2_normalize(dino.astype(np.float32))
    features = apply_mask_prototype_blend(
        features,
        mask_ids,
        blend=float(np.clip(args.prototype_blend, 0.0, 1.0)),
        min_points=args.prototype_min_points,
    )
    features = l2_normalize(features.astype(np.float32))
    rgb, pca_mean, pca_basis = pca_to_rgb(
        features,
        valid,
        max_samples=args.max_pca_samples,
        seed=args.seed,
    )

    copy_viewer_sidecar_files(model_path, output_model)
    dst_ply = output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    write_modified_ply(
        ply,
        property_names,
        rgb,
        dst_ply,
        zero_sh_rest=not args.keep_sh_rest,
    )

    metadata = {
        "source_model": str(model_path),
        "source_ply": str(src_ply),
        "output_model": str(output_model),
        "output_ply": str(dst_ply),
        "iteration": args.iteration,
        "num_gaussians": int(dino.shape[0]),
        "valid_feature_gaussians": int(valid.sum()),
        "feature_dim": int(dino.shape[1]),
        "prototype_blend": float(args.prototype_blend),
        "prototype_min_points": int(args.prototype_min_points),
        "zero_sh_rest": not args.keep_sh_rest,
        "pca_mean": pca_mean.astype(float).tolist(),
        "pca_basis": pca_basis.astype(float).tolist(),
    }
    if mask_ids is not None:
        metadata["unique_mask_ids"] = int(len(np.unique(mask_ids[mask_ids >= 0])))

    metadata_path = output_model / "feature_viewer_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

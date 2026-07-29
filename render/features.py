import os
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import FEATURE_RASTER_CHANNELS, render_feature_field
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.mask_completion import complete_unassigned_labels


def default_render_root(model_path):
    return os.path.join(os.path.dirname(os.path.normpath(model_path)), "render")


def project_features(
    features,
    out_dim=FEATURE_RASTER_CHANNELS,
    method="pca",
    normalize=True,
    pca_fit_features=None,
):
    if features.dim() == 3 and features.shape[-1] == 1:
        features = features.squeeze(-1)
    features = features.detach().float()
    if features.numel() == 0:
        raise ValueError("Gaussian model does not contain feature vectors.")

    in_dim = features.shape[1]
    if method == "slice":
        projected = features[:, : min(in_dim, out_dim)]
        if projected.shape[1] < out_dim:
            pad = torch.zeros(features.shape[0], out_dim - projected.shape[1], device=features.device)
            projected = torch.cat([projected, pad], dim=1)
    elif method == "pca":
        if in_dim < out_dim:
            raise ValueError(f"PCA projection requires input dim >= {out_dim}, got {in_dim}.")
        fit_features = features if pca_fit_features is None else pca_fit_features.detach().float().to(features.device)
        fit_mean = fit_features.mean(dim=0, keepdim=True)
        centered_fit = fit_features - fit_mean
        cov = centered_fit.T @ centered_fit / max(1, fit_features.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        basis = eigvecs[:, order[:out_dim]]
        projected = (features - fit_mean) @ basis
    else:
        raise ValueError(f"Unknown projection method: {method}")

    if normalize:
        projected = torch.nn.functional.normalize(projected, p=2, dim=1, eps=1e-6)

    return projected.contiguous()


def build_balanced_object_pca_fit(features, mask_ids, min_gaussians=32):
    if mask_ids is None or mask_ids.numel() == 0:
        return None

    mask_ids = mask_ids.to(features.device).long()
    assigned = mask_ids >= 0
    if not assigned.any():
        return None

    prototypes = []
    counts = []
    for obj_id in torch.unique(mask_ids[assigned]):
        obj_mask = mask_ids == obj_id
        count = int(obj_mask.sum().item())
        if count < int(min_gaussians):
            continue
        prototypes.append(features[obj_mask].mean(dim=0))
        counts.append((int(obj_id.item()), count))

    if len(prototypes) < 2:
        return None

    return torch.stack(prototypes, dim=0)


def apply_mask_prototypes(features, mask_ids, blend=1.0):
    if mask_ids is None or mask_ids.numel() == 0:
        print("Warning: no mask_ids found; using raw Gaussian features.")
        return features

    mask_ids = mask_ids.to(features.device).long()
    assigned = mask_ids >= 0
    if not assigned.any():
        print("Warning: no assigned mask_ids found; using raw Gaussian features.")
        return features

    blend = float(np.clip(blend, 0.0, 1.0))
    smoothed = features.detach().clone()
    unique_ids = torch.unique(mask_ids[assigned])

    for obj_id in unique_ids:
        obj_mask = mask_ids == obj_id
        if obj_mask.sum() == 0:
            continue
        prototype = features[obj_mask].mean(dim=0, keepdim=True)
        smoothed[obj_mask] = blend * prototype + (1.0 - blend) * features[obj_mask]

    return smoothed


def build_rgb_projection(projected_features, max_pca_samples=200000):
    features = projected_features.detach().float().cpu().numpy()
    if features.shape[0] > max_pca_samples:
        sample_idx = np.linspace(0, features.shape[0] - 1, max_pca_samples, dtype=np.int64)
        fit = features[sample_idx]
    else:
        fit = features

    mean = fit.mean(axis=0, keepdims=True)
    centered = fit - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    projected = (features - mean) @ basis
    lo = np.percentile(projected, 1, axis=0)
    hi = np.percentile(projected, 99, axis=0)
    return {"mean": mean, "basis": basis, "lo": lo, "hi": hi}


def feature_map_to_rgb(feature_map, mode="pca", max_pca_samples=100000, normalize_channels=True,
                       rgb_projection=None):
    feature_hw = feature_map.detach().float().cpu().permute(1, 2, 0).numpy()
    h, w, c = feature_hw.shape
    flat = feature_hw.reshape(-1, c)
    valid = np.linalg.norm(flat, axis=1) > 1e-8

    if not valid.any():
        return np.zeros((h, w, 3), dtype=np.uint8), False

    if normalize_channels:
        denom = np.linalg.norm(flat, axis=1, keepdims=True)
        flat = flat / np.maximum(denom, 1e-8)
        feature_hw = flat.reshape(h, w, c)

    fixed_lohi = None
    if mode == "pca":
        if rgb_projection is not None:
            projected = (flat - rgb_projection["mean"]) @ rgb_projection["basis"]
            preview = projected.reshape(h, w, 3)
            fixed_lohi = (rgb_projection["lo"], rgb_projection["hi"])
        else:
            valid_features = flat[valid]
            if valid_features.shape[0] > max_pca_samples:
                sample_idx = np.linspace(0, valid_features.shape[0] - 1, max_pca_samples, dtype=np.int64)
                fit_features = valid_features[sample_idx]
            else:
                fit_features = valid_features

            mean = fit_features.mean(axis=0, keepdims=True)
            centered = fit_features - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:3].T
            projected = (flat - mean) @ basis
            preview = projected.reshape(h, w, 3)
    else:
        raise ValueError(f"Unknown preview mode: {mode}")

    if fixed_lohi is not None:
        lo, hi = fixed_lohi
    else:
        valid_preview = preview.reshape(-1, 3)[valid]
        lo = np.percentile(valid_preview, 1, axis=0)
        hi = np.percentile(valid_preview, 99, axis=0)
    preview = (preview - lo) / (hi - lo + 1e-6)
    preview = np.clip(preview, 0.0, 1.0)
    preview.reshape(-1, 3)[~valid] = 0.0
    return (preview * 255).astype(np.uint8), True


def build_label_map_from_npz(
    feature_field_dir,
    image_name,
    height,
    width,
    fill_unassigned=False,
    fill_iterations=16,
    fill_edge_threshold=0.35,
):
    if not feature_field_dir:
        return None

    candidates = [
        os.path.join(feature_field_dir, f"{image_name}.npz"),
        os.path.join(feature_field_dir, f"{os.path.splitext(image_name)[0]}.npz"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None

    data = np.load(path, allow_pickle=True)
    if "masks" not in data:
        return None

    masks = data["masks"].astype(bool)
    if masks.size == 0:
        return None

    areas = data["areas"] if "areas" in data else masks.reshape(masks.shape[0], -1).sum(axis=1)
    label_map = np.full(masks.shape[1:], -1, dtype=np.int32)

    # Large masks first, smaller masks later, so object parts can overwrite broad regions.
    for mask_idx in np.argsort(areas)[::-1]:
        label_map[masks[mask_idx]] = int(mask_idx)

    if fill_unassigned and "edge_map" in data:
        label_map = complete_unassigned_labels(
            label_map,
            max_iterations=fill_iterations,
            edge_map=data["edge_map"].astype(np.float32),
            edge_threshold=fill_edge_threshold,
        )

    if label_map.shape != (height, width):
        label_map = cv2.resize(label_map, (width, height), interpolation=cv2.INTER_NEAREST)
    return label_map


def load_global_label_map(render_root, iteration, view_index, image_name, height, width):
    binary_root = os.path.join(render_root, "global_mask_ids", f"ours_{iteration}", "per_object_binary")
    if not os.path.isdir(binary_root):
        return None

    image_stem = os.path.splitext(image_name)[0]
    candidates = [
        f"{view_index:05d}.png",
        f"{view_index:02d}.png",
        f"{image_stem}.png",
    ]
    label_map = np.full((height, width), -1, dtype=np.int32)
    found = False

    for mask_dir in sorted(os.listdir(binary_root)):
        if not mask_dir.startswith("mask_"):
            continue
        try:
            mask_id = int(mask_dir.split("_")[-1])
        except ValueError:
            continue

        path = None
        for candidate in candidates:
            candidate_path = os.path.join(binary_root, mask_dir, candidate)
            if os.path.exists(candidate_path):
                path = candidate_path
                break
        if path is None:
            continue

        mask = np.array(Image.open(path).convert("L")) > 0
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        label_map[mask] = mask_id
        found = True

    return label_map if found else None


def apply_mask_guidance(preview_rgb, label_map, blend=0.55, min_area=512):
    if label_map is None:
        return preview_rgb, False

    guided = preview_rgb.astype(np.float32).copy()
    original = preview_rgb.astype(np.float32)
    blend = float(np.clip(blend, 0.0, 1.0))
    applied = False

    for label in np.unique(label_map):
        if label < 0:
            continue
        mask = label_map == label
        area = int(mask.sum())
        if area < min_area:
            continue

        colors = original[mask]
        # Median is more robust than mean when raw feature splats bleed across boundaries.
        prototype_color = np.median(colors, axis=0)
        guided[mask] = (1.0 - blend) * original[mask] + blend * prototype_color
        applied = True

    return np.clip(guided, 0, 255).astype(np.uint8), applied


def build_object_rgb_palette(features, mask_ids, min_gaussians=32):
    if mask_ids is None or mask_ids.numel() == 0:
        return None

    device = features.device
    mask_ids = mask_ids.to(device).long()
    assigned = mask_ids >= 0
    if not assigned.any():
        return None

    obj_ids = []
    prototypes = []
    for obj_id in torch.unique(mask_ids[assigned]):
        obj_mask = mask_ids == obj_id
        if int(obj_mask.sum().item()) < int(min_gaussians):
            continue
        obj_ids.append(int(obj_id.item()))
        prototypes.append(features[obj_mask].mean(dim=0))

    if len(prototypes) < 2:
        return None

    prototypes = torch.stack(prototypes, dim=0)
    projected = project_features(
        prototypes,
        out_dim=3,
        method="pca",
        normalize=True,
        pca_fit_features=prototypes,
    ).detach().cpu().numpy()

    lo = np.percentile(projected, 1, axis=0)
    hi = np.percentile(projected, 99, axis=0)
    rgb = np.clip((projected - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)
    return {int(obj_id): rgb[idx] for idx, obj_id in enumerate(obj_ids)}


def apply_palette_guidance(preview_rgb, label_map, palette, blend=1.0, min_area=512):
    if label_map is None or not palette:
        return preview_rgb, False

    guided = preview_rgb.astype(np.float32).copy()
    blend = float(np.clip(blend, 0.0, 1.0))
    applied = False

    for label in np.unique(label_map):
        if label < 0 or int(label) not in palette:
            continue
        mask = label_map == label
        if int(mask.sum()) < min_area:
            continue
        prototype_color = np.asarray(palette[int(label)], dtype=np.float32)
        guided[mask] = (1.0 - blend) * guided[mask] + blend * prototype_color
        applied = True

    return np.clip(guided, 0, 255).astype(np.uint8), applied


def draw_label_boundaries(rgb, label_map, width=2, color=(0, 0, 0), min_area=512):
    if label_map is None or width <= 0:
        return rgb

    output = rgb.copy()
    kernel = np.ones((2 * int(width) + 1, 2 * int(width) + 1), dtype=np.uint8)
    boundary = np.zeros(label_map.shape, dtype=bool)

    for label in np.unique(label_map):
        if label < 0:
            continue
        mask = (label_map == label).astype(np.uint8)
        if int(mask.sum()) < int(min_area):
            continue
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary |= (dilated != eroded)

    output[boundary] = np.asarray(color, dtype=np.uint8)
    return output


def enhance_feature_preview(rgb, contrast=1.25, saturation=1.15, sharpness=0.35):
    image = rgb.astype(np.float32) / 255.0
    contrast = float(max(0.0, contrast))
    saturation = float(max(0.0, saturation))
    sharpness = float(max(0.0, sharpness))

    if contrast != 1.0:
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast + mean

    image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if saturation != 1.0:
        hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        image_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if sharpness > 0.0:
        blurred = cv2.GaussianBlur(image_u8, (0, 0), sigmaX=1.2)
        image_u8 = cv2.addWeighted(image_u8, 1.0 + sharpness, blurred, -sharpness, 0)

    return image_u8


def render_feature_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    projected_features,
    save_npy,
    max_views,
    render_root=None,
    preview_mode="pca",
    mask_guidance_source="feature_field",
    feature_field_dir="",
    mask_guidance_blend=0.55,
    mask_guidance_min_area=512,
    fill_unassigned_guidance=False,
    fill_unassigned_iterations=96,
    fill_unassigned_edge_threshold=1.0,
    normalize_rendered_features=True,
    object_rgb_palette=None,
    draw_boundaries=False,
    boundary_width=2,
    enhance_preview=True,
    feature_contrast=1.25,
    feature_saturation=1.15,
    feature_sharpness=0.35,
    rgb_projection=None,
):
    render_root = render_root or default_render_root(model_path)
    out_root = os.path.join(render_root, "pca_feature_field", name, f"ours_{iteration}")
    rgb_dir = os.path.join(out_root, "rgb_preview")
    raw_rgb_dir = os.path.join(out_root, "rgb_preview_raw")
    npy_dir = os.path.join(out_root, "npy")
    os.makedirs(rgb_dir, exist_ok=True)
    if mask_guidance_source != "none":
        os.makedirs(raw_rgb_dir, exist_ok=True)
    if save_npy:
        os.makedirs(npy_dir, exist_ok=True)

    metadata = {
        "iteration": int(iteration),
        "split": name,
        "preview_mode": preview_mode,
        "rgb_projection_fixed": rgb_projection is not None,
        "mask_guidance_source": mask_guidance_source,
        "mask_guidance_blend": float(mask_guidance_blend),
        "mask_guidance_min_area": int(mask_guidance_min_area),
        "fill_unassigned_guidance": bool(fill_unassigned_guidance),
        "normalize_rendered_features": bool(normalize_rendered_features),
        "draw_boundaries": bool(draw_boundaries),
        "boundary_width": int(boundary_width),
        "enhance_preview": bool(enhance_preview),
        "feature_contrast": float(feature_contrast),
        "feature_saturation": float(feature_saturation),
        "feature_sharpness": float(feature_sharpness),
        "object_rgb_palette": None if object_rgb_palette is None else {
            str(k): np.asarray(v, dtype=np.uint8).astype(int).tolist()
            for k, v in object_rgb_palette.items()
        },
    }
    with open(os.path.join(out_root, "feature_render_config.json"), "w") as f:
        import json
        json.dump(metadata, f, indent=2)

    selected_views = views[:max_views] if max_views > 0 else views
    for view_index, view in enumerate(tqdm(selected_views, desc=f"Rendering {name} features")):
        with torch.no_grad():
            result = render_feature_field(
                view,
                gaussians,
                pipeline,
                background,
                feature_tensor=projected_features,
            )
        feature_map = result["feature_map"]
        preview_rgb, has_signal = feature_map_to_rgb(
            feature_map,
            mode=preview_mode,
            normalize_channels=normalize_rendered_features,
            rgb_projection=rgb_projection,
        )
        if not has_signal:
            print(f"Warning: rendered feature map is blank for view {view.image_name}")

        output_rgb = preview_rgb
        if mask_guidance_source != "none":
            height, width = preview_rgb.shape[:2]
            label_map = None
            if mask_guidance_source == "feature_field":
                label_map = build_label_map_from_npz(
                    feature_field_dir,
                    view.image_name,
                    height,
                    width,
                    fill_unassigned=fill_unassigned_guidance,
                    fill_iterations=fill_unassigned_iterations,
                    fill_edge_threshold=fill_unassigned_edge_threshold,
                )
            elif mask_guidance_source == "global":
                label_map = load_global_label_map(render_root, iteration, view_index, view.image_name, height, width)
                if fill_unassigned_guidance:
                    label_map = complete_unassigned_labels(
                        label_map,
                        max_iterations=fill_unassigned_iterations,
                    )
            else:
                raise ValueError(f"Unknown mask guidance source: {mask_guidance_source}")

            if object_rgb_palette is not None and mask_guidance_source == "global":
                guided_rgb, applied = apply_palette_guidance(
                    preview_rgb,
                    label_map,
                    object_rgb_palette,
                    blend=mask_guidance_blend,
                    min_area=mask_guidance_min_area,
                )
            else:
                guided_rgb, applied = apply_mask_guidance(
                    preview_rgb,
                    label_map,
                    blend=mask_guidance_blend,
                    min_area=mask_guidance_min_area,
                )
            if applied:
                output_rgb = guided_rgb
            else:
                print(f"Warning: mask guidance was not applied for view {view.image_name}")
            if draw_boundaries:
                output_rgb = draw_label_boundaries(
                    output_rgb,
                    label_map,
                    width=boundary_width,
                    min_area=mask_guidance_min_area,
                )
            Image.fromarray(preview_rgb).save(os.path.join(raw_rgb_dir, f"{view.image_name}.png"))

        if enhance_preview:
            output_rgb = enhance_feature_preview(
                output_rgb,
                contrast=feature_contrast,
                saturation=feature_saturation,
                sharpness=feature_sharpness,
            )

        Image.fromarray(output_rgb).save(os.path.join(rgb_dir, f"{view.image_name}.png"))
        if save_npy:
            np.save(os.path.join(npy_dir, f"{view.image_name}.npy"), feature_map.detach().cpu().numpy().astype(np.float32))
        torch.cuda.empty_cache()


def main():
    parser = ArgumentParser(description="Render experimental low-dimensional Gaussian feature fields.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--projection", choices=["pca", "slice"], default="pca")
    parser.add_argument("--preview_mode", choices=["pca", "first3"], default="pca")
    parser.add_argument("--feature_source", choices=["dino", "mask_prototype"], default="dino",
                        help="Use raw Gaussian DINO features or object-wise global-mask prototype features.")
    parser.add_argument("--prototype_blend", type=float, default=1.0,
                        help="For --feature_source mask_prototype, 1.0 means fully object-wise; 0.0 keeps raw features.")
    parser.add_argument("--mask_guidance_source", choices=["feature_field", "global", "none"], default="feature_field",
                        help="Post-process RGB preview with MCM refined masks or global mask IDs. Keeps raw render in rgb_preview_raw.")
    parser.add_argument("--mask_guidance_blend", type=float, default=0.55,
                        help="Blend strength for mask-guided feature preview. Larger is more object-consistent.")
    parser.add_argument("--mask_guidance_min_area", type=int, default=512,
                        help="Ignore mask regions smaller than this many pixels during mask-guided preview.")
    parser.add_argument("--object_palette_guidance", action="store_true",
                        help="For global mask guidance, color each object by its DINO prototype PCA color instead of the rendered median color.")
    parser.add_argument("--draw_mask_boundaries", action="store_true",
                        help="Draw hard boundaries on mask-guided feature previews for object-wise qualitative figures.")
    parser.add_argument("--mask_boundary_width", type=int, default=2,
                        help="Boundary width in pixels for --draw_mask_boundaries.")
    parser.add_argument("--no_enhance_feature_preview", action="store_true",
                        help="Disable contrast/saturation/sharpness enhancement for rgb_preview.")
    parser.add_argument("--feature_contrast", type=float, default=1.25,
                        help="False-color preview contrast multiplier.")
    parser.add_argument("--feature_saturation", type=float, default=1.15,
                        help="False-color preview saturation multiplier.")
    parser.add_argument("--feature_sharpness", type=float, default=0.35,
                        help="Unsharp-mask strength for false-color feature previews.")
    parser.add_argument("--fill_unassigned_guidance", action="store_true",
                        help="Fill unlabeled mask-guidance pixels before applying object-wise PCA colors.")
    parser.add_argument("--fill_unassigned_iterations", type=int, default=96,
                        help="Maximum pixel-growth iterations for --fill_unassigned_guidance.")
    parser.add_argument("--fill_unassigned_edge_threshold", type=float, default=1.0,
                        help="Do not propagate feature-field guidance through edge responses above this value.")
    parser.add_argument("--balanced_object_pca", action="store_true",
                        help="Fit PCA on one prototype per global mask ID instead of weighting by Gaussian count.")
    parser.add_argument("--balanced_object_pca_min_gaussians", type=int, default=32,
                        help="Ignore global mask IDs with fewer Gaussians when fitting balanced object PCA.")
    parser.add_argument("--no_normalize_rendered_features", action="store_true",
                        help="Disable per-pixel channel normalization before RGB preview PCA/first3 visualization.")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--max_views", type=int, default=0, help="Render at most this many views per split. 0 means all views.")
    parser.add_argument("--render_root", type=str, default=None,
                        help="Output root for render artifacts. Default: dirname(model_path)/render")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    if not hasattr(gaussians, "_dino_features") or gaussians._dino_features.numel() == 0:
        raise ValueError("Loaded model does not contain _dino_features. Train with feature-field lifting first.")
    feature_norm = gaussians.get_dino_features.detach().float().norm(dim=1)
    nonzero_features = int((feature_norm > 1e-8).sum().item())
    if nonzero_features == 0:
        raise ValueError(
            "Loaded model contains DINO feature columns, but all values are zero. "
        )

    source_features = gaussians.get_dino_features
    if args.feature_source == "mask_prototype":
        source_features = apply_mask_prototypes(
            source_features,
            getattr(gaussians, "mask_ids", None),
            blend=args.prototype_blend,
        )

    pca_fit_features = None
    if args.balanced_object_pca and args.projection == "pca":
        pca_fit_features = build_balanced_object_pca_fit(
            source_features,
            getattr(gaussians, "mask_ids", None),
            min_gaussians=args.balanced_object_pca_min_gaussians,
        )

    projected_features = project_features(
        source_features,
        out_dim=FEATURE_RASTER_CHANNELS,
        method=args.projection,
        normalize=not args.no_normalize,
        pca_fit_features=pca_fit_features,
    )
    rgb_projection = None
    if args.preview_mode == "pca":
        rgb_projection = build_rgb_projection(projected_features)

    object_rgb_palette = None
    if args.object_palette_guidance:
        object_rgb_palette = build_object_rgb_palette(
            source_features,
            getattr(gaussians, "mask_ids", None),
            min_gaussians=args.balanced_object_pca_min_gaussians,
        )
        if object_rgb_palette is not None:
            print(f"Object palette guidance: {len(object_rgb_palette)} global IDs")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_root = args.render_root or default_render_root(dataset.model_path)
    os.makedirs(render_root, exist_ok=True)

    if not args.skip_train:
        render_feature_set(
            dataset.model_path,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipe,
            background,
            projected_features,
            args.save_npy,
            args.max_views,
            render_root,
            args.preview_mode,
            args.mask_guidance_source,
            getattr(dataset, "feature_field_dir", ""),
            args.mask_guidance_blend,
            args.mask_guidance_min_area,
            args.fill_unassigned_guidance,
            args.fill_unassigned_iterations,
            args.fill_unassigned_edge_threshold,
            not args.no_normalize_rendered_features,
            object_rgb_palette,
            args.draw_mask_boundaries,
            args.mask_boundary_width,
            not args.no_enhance_feature_preview,
            args.feature_contrast,
            args.feature_saturation,
            args.feature_sharpness,
            rgb_projection,
        )
    if not args.skip_test:
        render_feature_set(
            dataset.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipe,
            background,
            projected_features,
            args.save_npy,
            args.max_views,
            render_root,
            args.preview_mode,
            args.mask_guidance_source,
            getattr(dataset, "feature_field_dir", ""),
            args.mask_guidance_blend,
            args.mask_guidance_min_area,
            args.fill_unassigned_guidance,
            args.fill_unassigned_iterations,
            args.fill_unassigned_edge_threshold,
            not args.no_normalize_rendered_features,
            object_rgb_palette,
            args.draw_mask_boundaries,
            args.mask_boundary_width,
            not args.no_enhance_feature_preview,
            args.feature_contrast,
            args.feature_saturation,
            args.feature_sharpness,
            rgb_projection,
        )


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import colorsys
import cv2
import json
import shutil
from collections import defaultdict

from segmentation import load_feature_field_directory
from utils.feature_lifting import (
    apply_zbuffer_visibility,
    build_projection_depth_buffer,
    build_cross_view_matching_graph,
    project_points_to_camera,
    refine_matching_with_multiview_consensus,
)
from utils.mask_completion import complete_unassigned_labels


def mask_id_to_color(mask_id, max_ids=256):
    if mask_id < 0:
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)  # Gray for unassigned

    # Use golden ratio for better color distribution
    h = (mask_id * 0.618033988749895) % 1.0
    s = 0.7 + (mask_id % 3) * 0.1  # Higher saturation
    l = 0.5 + (mask_id % 4) * 0.08

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return torch.tensor([r, g, b], dtype=torch.float32)


def default_render_root(model_path):
    return os.path.join(os.path.dirname(os.path.normpath(model_path)), "render")


def ensure_rgb_uint8(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def draw_global_id_labels(image, id_map, ids, min_area=64):
    labeled = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for obj_id in ids:
        obj_mask = (id_map == int(obj_id)).astype(np.uint8)
        if obj_mask.sum() < min_area:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, 8)
        if num_labels <= 1:
            continue

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest, cv2.CC_STAT_AREA] < min_area:
            continue

        cx, cy = centroids[largest]
        text = str(int(obj_id))
        font_scale = max(0.35, min(0.65, image.shape[1] / 2200.0))
        thickness = max(1, int(round(font_scale * 2)))
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(cx - text_size[0] / 2)
        y = int(cy + text_size[1] / 2)
        cv2.putText(labeled, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(labeled, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return labeled


def quantize_rendered_mask(rendered_rgb, unique_ids, color_table, background_threshold=6):
    h, w = rendered_rgb.shape[:2]
    id_map = np.full((h, w), -1, dtype=np.int32)
    if len(unique_ids) == 0:
        return id_map

    colors = np.stack([color_table[int(obj_id)] for obj_id in unique_ids], axis=0).astype(np.float32)
    pixels = rendered_rgb.reshape(-1, 3).astype(np.float32)
    foreground = np.linalg.norm(pixels, axis=1) > background_threshold

    if foreground.any():
        diff = pixels[foreground, None, :] - colors[None, :, :]
        nearest = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        flat_ids = id_map.reshape(-1)
        flat_ids[np.where(foreground)[0]] = np.asarray(unique_ids, dtype=np.int32)[nearest]

    return id_map


def colorize_id_map(id_map, color_table):
    color = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    for obj_id, rgb in color_table.items():
        color[id_map == int(obj_id)] = rgb
    return color


def clean_id_map(id_map, min_component_area=256):
    if min_component_area <= 0:
        return id_map
    cleaned = np.full_like(id_map, -1)
    for obj_id in np.unique(id_map):
        if obj_id < 0:
            continue
        mask = (id_map == obj_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned[labels == label] = obj_id
    return cleaned


def save_global_mask_images(
    id_map,
    rgb_img,
    file_name,
    unique_ids_np,
    color_table,
    mask_color_path,
    labeled_path,
    overlay_path,
    per_object_path,
    min_component_area=256,
):
    id_map = clean_id_map(id_map, min_component_area)
    hard_mask = colorize_id_map(id_map, color_table)
    labeled = draw_global_id_labels(hard_mask, id_map, unique_ids_np, min_area=min_component_area)

    overlay = (0.55 * rgb_img + 0.45 * hard_mask).astype(np.uint8)
    overlay[id_map < 0] = rgb_img[id_map < 0]
    overlay = draw_global_id_labels(overlay, id_map, unique_ids_np, min_area=min_component_area)

    Image.fromarray(hard_mask).save(os.path.join(mask_color_path, file_name))
    Image.fromarray(labeled).save(os.path.join(labeled_path, file_name))
    Image.fromarray(overlay).save(os.path.join(overlay_path, file_name))

    for obj_id in unique_ids_np:
        obj_dir = os.path.join(per_object_path, f"mask_{int(obj_id):03d}")
        os.makedirs(obj_dir, exist_ok=True)
        binary = ((id_map == int(obj_id)).astype(np.uint8) * 255)
        Image.fromarray(binary, mode="L").save(os.path.join(obj_dir, file_name))


def camera_rgb_uint8(view):
    gt = view.original_image[0:3, :, :].detach().cpu()
    return (gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)


def mapping_json_path(model_path, iteration):
    return Path(model_path) / f"mask_id_mapping_iter_{iteration}.json"


def build_or_load_global_mapping(model_path, iteration, cameras, feature_field_dir, args):
    feature_field_views = load_feature_field_directory(Path(feature_field_dir))
    if not feature_field_views:
        raise ValueError(f"No feature-field .npz files found in {feature_field_dir}")

    camera_by_name = {camera.image_name: camera for camera in cameras}
    saved_path = mapping_json_path(model_path, iteration)
    if saved_path.exists():
        with open(saved_path) as f:
            payload = json.load(f)
        view_names = payload.get("view_image_names", [])
        local_to_global_map = {
            (int(item["view_idx"]), int(item["local_mask_id"])): int(item["global_mask_id"])
            for item in payload.get("local_to_global_map", [])
        }
        view_entries = []
        for view_idx, image_name in enumerate(view_names):
            view_data = feature_field_views.get(image_name)
            camera = camera_by_name.get(image_name)
            if view_data is not None and camera is not None:
                view_entries.append((view_idx, camera, view_data))
        global_count = int(payload.get("global_mask_count", 0))
        return local_to_global_map, global_count, view_entries, feature_field_views, "saved_mapping"

    view_entries = []
    for idx, camera in enumerate(cameras):
        view_data = feature_field_views.get(camera.image_name)
        if view_data is None or view_data.feature_dim == 0:
            continue
        view_entries.append((len(view_entries), camera, view_data))

    all_mask_descriptors = []
    for view_idx, (_, _, view_data) in enumerate(view_entries):
        features = view_data.features
        for local_mask_id in range(features.shape[0]):
            all_mask_descriptors.append({
                "view_idx": view_idx,
                "local_mask_id": local_mask_id,
                "feature": features[local_mask_id].cpu(),
            })

    same_view_threshold = (
        float(args.feature_field_same_view_matching_threshold)
        if float(args.feature_field_same_view_matching_threshold) >= 0
        else None
    )
    local_to_global_map, global_count = build_cross_view_matching_graph(
        all_mask_descriptors,
        similarity_threshold=float(args.feature_field_matching_threshold),
        same_view_threshold=same_view_threshold,
        max_view_gap=int(args.feature_field_matching_max_view_gap),
        topk_per_mask=int(args.feature_field_matching_topk_per_mask),
        one_to_one_per_view_pair=bool(args.feature_field_matching_one_to_one),
    )
    if bool(args.use_multiview_refinement):
        local_to_global_map, global_count = refine_matching_with_multiview_consensus(
            local_to_global_map,
            all_mask_descriptors,
            view_entries,
            min_consensus_views=int(args.min_consensus_views),
        )
    return local_to_global_map, global_count, view_entries, feature_field_views, "rebuilt_mapping"


def assign_local_masks_from_projected_gaussians(
    gaussians,
    camera,
    view_data,
    mask_ids,
    valid_global_ids,
    fallback_local_to_global,
    view_idx,
    min_votes=1,
    min_ratio=0.0,
    allow_fallback=False,
    use_zbuffer_visibility=True,
    zbuffer_abs_tolerance=0.02,
    zbuffer_rel_tolerance=0.03,
):
    mask_indices = view_data.mask_indices.to(mask_ids.device).long()
    num_local_masks = int(view_data.features.shape[0])
    if num_local_masks == 0:
        return {}, {"projected_points": 0, "majority_assigned": 0, "fallback_assigned": 0}, {}

    with torch.no_grad():
        depth_buffer = None
        if use_zbuffer_visibility:
            depth_buffer = build_projection_depth_buffer(
                gaussians.get_xyz,
                camera,
                view_data.width,
                view_data.height,
            )

        points_2d, depths, valid = project_points_to_camera(gaussians.get_xyz, camera)
        valid = apply_zbuffer_visibility(
            points_2d,
            depths,
            valid,
            depth_buffer,
            view_data.width,
            view_data.height,
            abs_tolerance=zbuffer_abs_tolerance,
            rel_tolerance=zbuffer_rel_tolerance,
        )
        xs = ((points_2d[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5) * (view_data.width - 1)
        ys = ((points_2d[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (view_data.height - 1)
        xi = torch.clamp(xs.round().long(), 0, view_data.width - 1)
        yi = torch.clamp(ys.round().long(), 0, view_data.height - 1)

        local_ids = mask_indices[yi, xi]
        valid = valid & (local_ids >= 0) & (mask_ids >= 0)
        if valid_global_ids:
            valid_ids_tensor = torch.tensor(sorted(valid_global_ids), device=mask_ids.device, dtype=mask_ids.dtype)
            valid = valid & torch.isin(mask_ids, valid_ids_tensor)

        local_ids_np = local_ids[valid].detach().cpu().numpy().astype(np.int64)
        global_ids_np = mask_ids[valid].detach().cpu().numpy().astype(np.int64)

    local_to_global = {}
    local_assignment_info = {}
    majority_assigned = 0
    fallback_assigned = 0
    for local_id in range(num_local_masks):
        selected = global_ids_np[local_ids_np == local_id]
        if selected.size > 0:
            ids, counts = np.unique(selected, return_counts=True)
            best_idx = int(np.argmax(counts))
            best_count = int(counts[best_idx])
            ratio = best_count / max(1, int(selected.size))
            if best_count >= min_votes and ratio >= min_ratio:
                local_to_global[local_id] = int(ids[best_idx])
                local_assignment_info[local_id] = {
                    "global_mask_id": int(ids[best_idx]),
                    "source": "projected_3d_majority",
                    "vote_count": best_count,
                    "vote_ratio": float(ratio),
                    "projected_points": int(selected.size),
                }
                majority_assigned += 1
                continue

        fallback_id = fallback_local_to_global.get((int(view_idx), int(local_id)), -1)
        if allow_fallback and fallback_id in valid_global_ids:
            local_to_global[local_id] = int(fallback_id)
            local_assignment_info[local_id] = {
                "global_mask_id": int(fallback_id),
                "source": "cross_view_mapping_fallback",
                "vote_count": int(selected.size),
                "vote_ratio": 0.0,
                "projected_points": int(selected.size),
            }
            fallback_assigned += 1

    return local_to_global, {
        "projected_points": int(global_ids_np.size),
        "majority_assigned": int(majority_assigned),
        "fallback_assigned": int(fallback_assigned),
    }, local_assignment_info


def project_gaussian_mask_ids_to_image(gaussians, camera, mask_ids, valid_global_ids, height, width):
    with torch.no_grad():
        points_2d, depths, valid = project_points_to_camera(gaussians.get_xyz, camera)
        valid = valid & (mask_ids >= 0)
        if valid_global_ids:
            valid_ids_tensor = torch.tensor(sorted(valid_global_ids), device=mask_ids.device, dtype=mask_ids.dtype)
            valid = valid & torch.isin(mask_ids, valid_ids_tensor)
        if not valid.any():
            return np.full((height, width), -1, dtype=np.int32)

        points_2d = points_2d[valid]
        depths = depths[valid]
        ids = mask_ids[valid]
        xs = ((points_2d[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5) * (width - 1)
        ys = ((points_2d[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (height - 1)
        xi = torch.clamp(xs.round().long(), 0, width - 1).detach().cpu().numpy()
        yi = torch.clamp(ys.round().long(), 0, height - 1).detach().cpu().numpy()
        depths_np = depths.detach().cpu().numpy()
        ids_np = ids.detach().cpu().numpy().astype(np.int32)

    sparse = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    order = np.argsort(depths_np)
    for idx in order:
        y = int(yi[idx])
        x = int(xi[idx])
        z = float(depths_np[idx])
        if z < depth_buffer[y, x]:
            depth_buffer[y, x] = z
            sparse[y, x] = int(ids_np[idx])
    return sparse


def render_refined_2d_global_masks(
    model_path,
    iteration,
    views,
    gaussians,
    feature_field_dir,
    args,
    render_root,
    color_table,
    unique_ids_np,
    output_dirs,
):
    local_to_global_map, global_count, view_entries, _, mapping_source = build_or_load_global_mapping(
        model_path, iteration, views, feature_field_dir, args
    )
    mask_color_path, labeled_path, overlay_path, per_object_path = output_dirs

    view_to_global_counts = {}
    view_projection_stats = {}
    per_view_assignments = {}
    object_pixel_area = defaultdict(int)
    object_view_names = defaultdict(set)
    global_id_source = getattr(args, "global_id_source", "mapping")
    if global_id_source not in {"mapping", "3d_majority"}:
        raise ValueError(f"Unknown --global_id_source: {global_id_source}")

    mapped_global_ids = set(int(x) for x in local_to_global_map.values())
    point_cloud_global_ids = set(int(x) for x in unique_ids_np)
    if global_id_source == "mapping":
        include_mapping_only_ids = bool(getattr(args, "include_mapping_only_global_ids", False))
        valid_global_ids = mapped_global_ids if include_mapping_only_ids else (mapped_global_ids & point_cloud_global_ids)
        for global_id in sorted(valid_global_ids):
            if global_id not in color_table:
                color_table[global_id] = (mask_id_to_color(global_id, max(global_count, 1)).numpy() * 255).astype(np.uint8)
            if global_id not in unique_ids_np:
                unique_ids_np.append(global_id)
        unique_ids_np.sort()
    else:
        valid_global_ids = point_cloud_global_ids

    mask_ids = gaussians.mask_ids.to("cuda")
    for idx, (view_idx, camera, view_data) in enumerate(tqdm(view_entries, desc="Rendering 2D global masks")):
        local_map = view_data.mask_indices.detach().cpu().numpy().astype(np.int32)
        id_map = np.full_like(local_map, -1, dtype=np.int32)
        if global_id_source == "3d_majority":
            projected_local_to_global, projection_stats, local_assignment_info = assign_local_masks_from_projected_gaussians(
                gaussians,
                camera,
                view_data,
                mask_ids,
                valid_global_ids,
                local_to_global_map,
                view_idx,
                min_votes=int(getattr(args, "global_id_majority_min_votes", 1)),
                min_ratio=float(getattr(args, "global_id_majority_min_ratio", 0.0)),
                allow_fallback=bool(getattr(args, "allow_3d_majority_mapping_fallback", False)),
                use_zbuffer_visibility=not bool(getattr(args, "disable_global_id_zbuffer_visibility", False)),
                zbuffer_abs_tolerance=float(getattr(args, "global_id_zbuffer_abs_tolerance", 0.02)),
                zbuffer_rel_tolerance=float(getattr(args, "global_id_zbuffer_rel_tolerance", 0.03)),
            )
        else:
            projected_local_to_global = {
                int(local_id): int(global_id)
                for (src_view_idx, local_id), global_id in local_to_global_map.items()
                if int(src_view_idx) == int(view_idx)
            }
            projection_stats = {
                "projected_points": 0,
                "majority_assigned": 0,
                "fallback_assigned": 0,
                "mapping_assigned": int(len(projected_local_to_global)),
            }
            local_assignment_info = {
                int(local_id): {
                    "global_mask_id": int(global_id),
                    "source": "cross_view_mapping",
                    "vote_count": 0,
                    "vote_ratio": 1.0,
                    "projected_points": 0,
                }
                for local_id, global_id in projected_local_to_global.items()
            }
        view_assignments = []
        for local_id in np.unique(local_map):
            if local_id < 0:
                continue
            global_id = projected_local_to_global.get(int(local_id), -1)
            if global_id in valid_global_ids:
                local_pixels = local_map == int(local_id)
                id_map[local_pixels] = int(global_id)
                info = local_assignment_info.get(int(local_id), {})
                area = int(local_pixels.sum())
                view_assignments.append({
                    "local_mask_id": int(local_id),
                    "global_mask_id": int(global_id),
                    "pixel_area": area,
                    "assignment_source": info.get("source", "unknown"),
                    "vote_count": int(info.get("vote_count", 0)),
                    "vote_ratio": float(info.get("vote_ratio", 0.0)),
                    "projected_points": int(info.get("projected_points", 0)),
                })
                object_pixel_area[int(global_id)] += area
                object_view_names[int(global_id)].add(camera.image_name)

        if bool(getattr(args, "fill_unassigned_global_masks", False)):
            projected_id_map = project_gaussian_mask_ids_to_image(
                gaussians,
                camera,
                mask_ids,
                valid_global_ids,
                id_map.shape[0],
                id_map.shape[1],
            )
            hole = (id_map < 0) & (projected_id_map >= 0)
            id_map[hole] = projected_id_map[hole]
            edge_map = view_data.edge_map.detach().cpu().numpy()
            id_map = complete_unassigned_labels(
                id_map,
                max_iterations=int(getattr(args, "fill_unassigned_iterations", 16)),
                edge_map=edge_map,
                edge_threshold=float(getattr(args, "fill_unassigned_edge_threshold", 0.35)),
            )

        rgb_img = camera_rgb_uint8(camera)
        target_h, target_w = rgb_img.shape[:2]
        if id_map.shape[:2] != (target_h, target_w):
            id_map = cv2.resize(id_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        file_name = f"{idx:05d}.png"
        save_global_mask_images(
            id_map,
            rgb_img,
            file_name,
            unique_ids_np,
            color_table,
            mask_color_path,
            labeled_path,
            overlay_path,
            per_object_path,
            min_component_area=int(args.mask_min_component_area),
        )
        view_to_global_counts[camera.image_name] = int(len(np.unique(id_map[id_map >= 0])))
        view_projection_stats[camera.image_name] = projection_stats
        per_view_assignments[camera.image_name] = {
            "view_index": int(idx),
            "source_view_idx": int(view_idx),
            "file_name": file_name,
            "global_id_mask": os.path.join(mask_color_path, file_name),
            "global_id_overlay": os.path.join(overlay_path, file_name),
            "assignments": sorted(view_assignments, key=lambda item: item["local_mask_id"]),
        }

    return {
        "source": (
            "refined_2d_feature_field_masks_with_cross_view_mapping"
            if global_id_source == "mapping"
            else "refined_2d_feature_field_masks_with_3d_majority_ids"
        ),
        "global_id_source": global_id_source,
        "mapping_source": mapping_source,
        "feature_field_dir": str(feature_field_dir),
        "global_mask_count": int(global_count),
        "rendered_global_mask_count": int(len(valid_global_ids)),
        "mapping_only_global_ids_hidden": bool(
            global_id_source == "mapping"
            and not bool(getattr(args, "include_mapping_only_global_ids", False))
        ),
        "dropped_mapping_only_global_ids": sorted(
            int(x) for x in (mapped_global_ids - point_cloud_global_ids)
        ),
        "mapped_global_ids": sorted(int(x) for x in mapped_global_ids),
        "point_cloud_global_ids": sorted(int(x) for x in point_cloud_global_ids),
        "view_global_id_counts": view_to_global_counts,
        "view_projection_stats": view_projection_stats,
        "per_view_assignments": per_view_assignments,
        "object_view_counts": {str(k): len(v) for k, v in object_view_names.items()},
        "object_pixel_area": {str(k): int(v) for k, v in object_pixel_area.items()},
    }


def render_mask_visualization(model_path, iteration, views, gaussians, pipeline, background, render_root=None, args=None):
    if not hasattr(gaussians, 'mask_ids'):
        return

    mask_ids = gaussians.mask_ids
    unique_ids = torch.unique(mask_ids[mask_ids >= 0])
    num_unique = len(unique_ids)

    min_editable_gaussians = int(getattr(args, "min_editable_gaussians", 256)) if args is not None else 256
    render_root = render_root or default_render_root(model_path)
    base_path = os.path.join(render_root, "global_mask_ids", f"ours_{iteration}")
    soft_color_path = os.path.join(base_path, "soft_color")
    mask_color_path = os.path.join(base_path, "global_id_masks")
    labeled_path = os.path.join(base_path, "global_id_labeled")
    overlay_path = os.path.join(base_path, "global_id_overlays")
    per_object_path = os.path.join(base_path, "per_object_binary")

    # Generate color map
    colors = torch.zeros(len(mask_ids), 3, device='cuda')
    unique_ids_np = [int(x) for x in unique_ids.cpu().numpy()]
    color_table = {
        int(mid): (mask_id_to_color(int(mid), num_unique).numpy() * 255).astype(np.uint8)
        for mid in unique_ids_np
    }
    for gid in range(len(mask_ids)):
        mid = int(mask_ids[gid].item())
        if mid >= 0:
            colors[gid] = mask_id_to_color(mid, num_unique).cuda()

    output_dirs = (mask_color_path, labeled_path, overlay_path, per_object_path)
    use_splat_global_masks = bool(getattr(args, "use_splat_global_masks", False)) if args is not None else False
    feature_field_dir = getattr(args, "feature_field_dir", "") if args is not None else ""
    if (not use_splat_global_masks) and feature_field_dir and os.path.isdir(feature_field_dir):
        for path in [mask_color_path, labeled_path, overlay_path, per_object_path]:
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        metadata_extra = render_refined_2d_global_masks(
            model_path,
            iteration,
            views,
            gaussians,
            feature_field_dir,
            args,
            render_root,
            color_table,
            unique_ids_np,
            output_dirs,
        )
        metadata = {
            "iteration": int(iteration),
            "num_gaussians": int(len(mask_ids)),
            "assigned_gaussians": int((mask_ids >= 0).sum().item()),
            "num_global_mask_ids": int(len(unique_ids_np)),
            "rendering_mode": "crisp_refined_2d_masks_with_cross_view_global_ids",
            "editing": {
                "point_cloud_path": os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"),
                "mask_property": "mask_id",
                "query_template": "mask_id == <global_mask_id>",
                "min_editable_gaussians": int(min_editable_gaussians),
            },
            "stable_color_table": {
                str(mid): color_table[int(mid)].astype(int).tolist()
                for mid in unique_ids_np
            },
            "objects": [],
            **metadata_extra,
        }
        object_view_counts = metadata_extra.get("object_view_counts", {})
        object_pixel_area = metadata_extra.get("object_pixel_area", {})
        for mid in unique_ids_np:
            num_obj_gaussians = int((mask_ids == int(mid)).sum().item())
            view_count = int(object_view_counts.get(str(mid), 0))
            metadata["objects"].append({
                "global_mask_id": int(mid),
                "num_gaussians": num_obj_gaussians,
                "view_count": view_count,
                "total_2d_pixel_area": int(object_pixel_area.get(str(mid), 0)),
                "rgb": color_table[int(mid)].astype(int).tolist(),
                "point_cloud_access": f"mask_id == {int(mid)}",
                "editable": bool(num_obj_gaussians >= min_editable_gaussians),
            })
        metadata["objects"] = sorted(
            metadata["objects"],
            key=lambda item: item["num_gaussians"],
            reverse=True,
        )
        metadata["editable_objects"] = [
            item for item in metadata["objects"] if item["editable"]
        ]
        with open(os.path.join(base_path, "global_mask_index.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(base_path, "view_consistent_id_index.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(base_path, "per_view_local_to_global.json"), "w") as f:
            json.dump(metadata.get("per_view_assignments", {}), f, indent=2)
        write_legend(unique_ids, color_table, base_path)
        return

    for path in [soft_color_path, mask_color_path, labeled_path, overlay_path, per_object_path]:
        os.makedirs(path, exist_ok=True)

    # Temporarily replace Gaussian colors
    original_features_dc = gaussians._features_dc.clone()
    original_opacity = gaussians._opacity.clone()
    with torch.no_grad():
        gaussians._features_dc.copy_(colors.unsqueeze(1))  # (N, 1, 3)
        gaussians._opacity[mask_ids < 0] = -1e10

    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        try:
            black_background = torch.zeros(3, dtype=torch.float32, device="cuda")
            rendering = render(view, gaussians, pipeline, black_background)["render"]
            soft_img = rendering.detach().cpu().permute(1, 2, 0).numpy()
            soft_img = (np.clip(soft_img, 0, 1) * 255).astype(np.uint8)

            with torch.no_grad():
                gaussians._features_dc.copy_(original_features_dc)
                gaussians._opacity.copy_(original_opacity)
            rgb_render = render(view, gaussians, pipeline, background)["render"]
            rgb_img = rgb_render.detach().cpu().permute(1, 2, 0).numpy()
            rgb_img = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
            with torch.no_grad():
                gaussians._features_dc.copy_(colors.unsqueeze(1))
                gaussians._opacity[mask_ids < 0] = -1e10

            id_map = quantize_rendered_mask(soft_img, unique_ids_np, color_table)
            file_name = f"{idx:05d}.png"
            Image.fromarray(soft_img).save(os.path.join(soft_color_path, file_name))
            save_global_mask_images(
                id_map,
                rgb_img,
                file_name,
                unique_ids_np,
                color_table,
                mask_color_path,
                labeled_path,
                overlay_path,
                per_object_path,
                min_component_area=int(getattr(args, "mask_min_component_area", 256)) if args is not None else 256,
            )
        except Exception as e:
            continue

    # Restore original colors
    with torch.no_grad():
        gaussians._features_dc.copy_(original_features_dc)
        gaussians._opacity.copy_(original_opacity)

    metadata = {
        "iteration": int(iteration),
        "num_gaussians": int(len(mask_ids)),
        "assigned_gaussians": int((mask_ids >= 0).sum().item()),
        "num_global_mask_ids": int(num_unique),
        "rendering_mode": "splat_color_quantization_debug",
        "objects": []
    }
    for mid in unique_ids_np:
        metadata["objects"].append({
            "global_mask_id": int(mid),
            "num_gaussians": int((mask_ids == int(mid)).sum().item()),
            "rgb": color_table[int(mid)].astype(int).tolist(),
            "point_cloud_access": f"mask_id == {int(mid)}",
        })
    with open(os.path.join(base_path, "global_mask_index.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        

def render_object_removal(model_path, iteration, views, gaussians, pipeline, background, remove_ids, render_root=None):
    if not hasattr(gaussians, 'mask_ids'):
        return

    mask_ids = gaussians.mask_ids
    remove_ids = set(remove_ids)

    # Create mask for Gaussians to keep
    keep_mask = torch.ones(len(mask_ids), dtype=torch.bool, device='cuda')
    for rid in remove_ids:
        keep_mask &= (mask_ids != rid)

    render_root = render_root or default_render_root(model_path)
    render_path = os.path.join(render_root, f"object_removal_{'_'.join(map(str, sorted(remove_ids)))}", f"ours_{iteration}")
    os.makedirs(render_path, exist_ok=True)

    # Set removed Gaussians to zero opacity
    original_opacity = gaussians._opacity.clone()

    with torch.no_grad():
        gaussians._opacity[~keep_mask] = -1e10  # Very low opacity

    edited_ply_path = os.path.join(render_path, "point_cloud.removed.ply")
    gaussians.save_ply(edited_ply_path)
    print(f"Edited PLY saved: {edited_ply_path}")

    print(f"Rendering {len(views)} views with objects removed.")
    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        img = rendering.detach().cpu().permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(render_path, f'{view.image_name}.png'))

    # Restore opacity
    with torch.no_grad():
        gaussians._opacity.copy_(original_opacity)
    print(f"Object removal complete: {render_path}")


def render_object_isolation(model_path, iteration, views, gaussians, pipeline, background, keep_ids, render_root=None):
    if not hasattr(gaussians, 'mask_ids'):
        return

    mask_ids = gaussians.mask_ids
    keep_ids = set(keep_ids)

    # Create mask for Gaussians to keep
    keep_mask = torch.zeros(len(mask_ids), dtype=torch.bool, device='cuda')
    for kid in keep_ids:
        keep_mask |= (mask_ids == kid)

    render_root = render_root or default_render_root(model_path)
    render_path = os.path.join(render_root, f"object_isolation_{'_'.join(map(str, sorted(keep_ids)))}", f"ours_{iteration}")
    os.makedirs(render_path, exist_ok=True)

    # Set non-kept Gaussians to zero opacity
    original_opacity = gaussians._opacity.clone()

    with torch.no_grad():
        gaussians._opacity[~keep_mask] = -1e10

    edited_ply_path = os.path.join(render_path, "point_cloud.isolated.ply")
    gaussians.save_ply(edited_ply_path)
    print(f"Isolated PLY saved: {edited_ply_path}")

    print(f"Rendering {len(views)} views with isolated objects")
    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        img = rendering.detach().cpu().permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(render_path, f'{view.image_name}.png'))

    # Restore opacity
    with torch.no_grad():
        gaussians._opacity.copy_(original_opacity)
    print(f"Object isolation complete: {render_path}")


def render_object_masks(model_path, iteration, views, gaussians, pipeline, background, render_root=None):
    if not hasattr(gaussians, 'mask_ids'):
        return

    mask_ids = gaussians.mask_ids
    unique_ids = torch.unique(mask_ids[mask_ids >= 0])

    render_root = render_root or default_render_root(model_path)
    masks_path = os.path.join(render_root, "object_masks", f"ours_{iteration}")
    os.makedirs(masks_path, exist_ok=True)

    original_opacity = gaussians._opacity.clone()

    for obj_id in tqdm(unique_ids.cpu().numpy(), desc="Processing objects"):
        obj_path = os.path.join(masks_path, f"mask_{int(obj_id):03d}")
        os.makedirs(obj_path, exist_ok=True)

        # Keep only this object
        keep_mask = (mask_ids == obj_id)

        with torch.no_grad():
            gaussians._opacity[~keep_mask] = -1e10

        for view in views:
            rendering = render(view, gaussians, pipeline, background)["render"]
            # Convert to grayscale mask
            img = rendering.detach().cpu().permute(1, 2, 0).mean(dim=2).numpy()
            img = (img > 0.01).astype(np.uint8) * 255
            Image.fromarray(img, mode='L').save(os.path.join(obj_path, f'{view.image_name}.png'))

        # Restore opacity
        with torch.no_grad():
            gaussians._opacity.copy_(original_opacity)

    print(f"Object masks complete: {masks_path}")


def load_mask_ids_from_config(config_file):
    if not config_file:
        return []

    with open(config_file, "r") as f:
        config = json.load(f)

    for key in ("select_obj_id", "mask_ids", "global_mask_ids"):
        if key in config:
            value = config[key]
            if isinstance(value, int):
                return [int(value)]
            return [int(x) for x in value]

    raise ValueError(
        f"No object IDs found in {config_file}. "
    )


def main():
    parser = ArgumentParser(description="Render and edit objects based on 3D mask IDs")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--mode', type=str, default='visualize',
                       choices=['visualize', 'remove', 'isolate', 'masks', 'all'],
                       help='Operation mode')
    parser.add_argument('--mask_ids', type=int, nargs='+', default=[],
                       help='Mask IDs to remove/isolate')
    parser.add_argument('--config_file', type=str, default="",
                       help='Optional GaussianGrouping-style JSON config with select_obj_id/mask_ids/global_mask_ids.')
    parser.add_argument('--render_root', type=str, default=None,
                       help='Output root for render artifacts. Default: dirname(model_path)/render')
    parser.add_argument('--use_splat_global_masks', action='store_true',
                       help='Debug mode: render global masks by coloring Gaussian splats instead of using crisp refined 2D masks.')
    parser.add_argument('--global_id_source', choices=['mapping', '3d_majority'], default='mapping',
                       help='mapping preserves refined 2D mask boundaries and recolors by cross-view ID; 3d_majority aligns each 2D mask to projected Gaussian IDs.')
    parser.add_argument('--global_id_majority_min_votes', type=int, default=1,
                       help='Minimum projected Gaussian votes required to assign a refined 2D mask in --global_id_source 3d_majority.')
    parser.add_argument('--global_id_majority_min_ratio', type=float, default=0.0,
                       help='Minimum majority vote ratio required to assign a refined 2D mask in --global_id_source 3d_majority.')
    parser.add_argument('--allow_3d_majority_mapping_fallback', action='store_true',
                       help='Debug mode: assign cross-view mapping IDs to refined masks without enough projected 3D votes.')
    parser.add_argument('--disable_global_id_zbuffer_visibility', action='store_true',
                       help='Debug mode: disable z-buffer visibility gating for projected 3D majority global IDs.')
    parser.add_argument('--global_id_zbuffer_abs_tolerance', type=float, default=0.02,
                       help='Absolute depth tolerance for z-buffer visibility in --global_id_source 3d_majority.')
    parser.add_argument('--global_id_zbuffer_rel_tolerance', type=float, default=0.03,
                       help='Relative depth tolerance for z-buffer visibility in --global_id_source 3d_majority.')
    parser.add_argument('--include_mapping_only_global_ids', action='store_true',
                       help='Debug mode: include cross-view 2D IDs that have no assigned 3D Gaussians in legends and masks.')
    parser.add_argument('--mask_min_component_area', type=int, default=256,
                       help='Remove connected components smaller than this area from saved global mask images.')
    parser.add_argument('--min_editable_gaussians', type=int, default=256,
                       help='Minimum Gaussians for a global mask ID to be marked editable in exported JSON.')
    parser.add_argument('--fill_unassigned_global_masks', action='store_true',
                       help='Fill unlabeled pixels in crisp global mask outputs by bounded neighbor propagation.')
    parser.add_argument('--fill_unassigned_iterations', type=int, default=96,
                       help='Maximum pixel-growth iterations for --fill_unassigned_global_masks.')
    parser.add_argument('--fill_unassigned_edge_threshold', type=float, default=1.0,
                       help='Do not propagate labels through pixels whose feature-field edge response exceeds this value.')
    args = get_combined_args(parser)
    if not args.mask_ids and args.config_file:
        args.mask_ids = load_mask_ids_from_config(args.config_file)

    # safe_state(args.quiet)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    render_root = args.render_root or default_render_root(args.model_path)
    os.makedirs(render_root, exist_ok=True)
    print(f"Render root: {render_root}")

    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not args.skip_train:
        views = scene.getTrainCameras()
        if args.mode == 'visualize' or args.mode == 'all':
            render_mask_visualization(args.model_path, scene.loaded_iter, views,
                                     gaussians, pp.extract(args), background, render_root, args)

        if args.mode == 'remove' and args.mask_ids:
            render_object_removal(args.model_path, scene.loaded_iter, views,
                                 gaussians, pp.extract(args), background, args.mask_ids, render_root)

        if args.mode == 'isolate' and args.mask_ids:
            render_object_isolation(args.model_path, scene.loaded_iter, views,
                                   gaussians, pp.extract(args), background, args.mask_ids, render_root)

        if args.mode == 'masks' or args.mode == 'all':
            render_object_masks(args.model_path, scene.loaded_iter, views,
                               gaussians, pp.extract(args), background, render_root)


if __name__ == "__main__":
    main()

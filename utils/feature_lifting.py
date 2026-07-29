from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from segmentation.mask_feature_dataset import MaskFeatureViewData
from collections import defaultdict


def project_points_to_camera(points_3d, camera):
    # World to camera transformation
    R = camera.R  # (3, 3)
    T = camera.T  # (3,)

    # Ensure R and T are torch tensors on GPU
    if not isinstance(R, torch.Tensor):
        R = torch.from_numpy(R).float().cuda()
    else:
        R = R.cuda()

    if not isinstance(T, torch.Tensor):
        T = torch.from_numpy(T).float().cuda()
    else:
        T = T.cuda()

    # Transform to camera space
    # points_cam = R @ points_3d.T + T.reshape(3, 1)
    points_cam = torch.matmul(points_3d, R.T) + T  # (N, 3)

    # Extract depth (z-coordinate in camera space)
    depths = points_cam[:, 2]  # (N,)

    # Projection matrix
    proj_matrix = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=camera.FoVx,
        fovY=camera.FoVy
    ).cuda()  # (4, 4)

    # Homogeneous coordinates
    points_cam_h = torch.cat([points_cam, torch.ones(points_cam.shape[0], 1).cuda()], dim=1)  # (N, 4)

    # Project to clip space
    points_clip = torch.matmul(points_cam_h, proj_matrix.T)  # (N, 4)

    # Perspective divide
    points_ndc = points_clip[:, :2] / points_clip[:, 3:4]  # (N, 2)
    points_2d = points_ndc

    # Visible points must be in front of the camera and inside the image.
    in_bounds = (
        (points_2d[:, 0] >= -1.0)
        & (points_2d[:, 0] <= 1.0)
        & (points_2d[:, 1] >= -1.0)
        & (points_2d[:, 1] <= 1.0)
    )
    valid_mask = (depths > 0.01) & in_bounds

    return points_2d, depths, valid_mask


def compute_visibility_weights(points_3d, camera, depths, valid_mask):
    
    N = points_3d.shape[0]
    weights = torch.zeros(N).cuda()

    # 1. Visibility
    weights[~valid_mask] = 0.0

    if valid_mask.sum() == 0:
        return weights

    # 2. Depth-based weight
    # Inverse depth with normalization
    valid_depths = depths[valid_mask]
    depth_weights = 1.0 / (valid_depths + 1e-6)
    depth_weights = depth_weights / (depth_weights.max() + 1e-6)  # Normalize to [0, 1]

    # 3. Viewing angle weight
    # Camera direction in world space
    camera_center = camera.camera_center  # (3,) world position of camera

    if not isinstance(camera_center, torch.Tensor):
        camera_center = torch.from_numpy(camera_center).float().cuda()
    else:
        camera_center = camera_center.cuda()

    view_directions = points_3d[valid_mask] - camera_center  # (N_valid, 3)
    view_directions = F.normalize(view_directions, p=2, dim=1)

    # Camera's forward direction (looking down -Z in camera space)
    R = camera.R
    if not isinstance(R, torch.Tensor):
        R = torch.from_numpy(R).float().cuda()
    camera_forward = -R[2, :]  # (3,) third row of R matrix

    # Cosine similarity: 1 = forward, 0 = side, -1 = back
    cos_angles = torch.sum(view_directions * camera_forward, dim=1)  # (N_valid,)
    angle_weights = torch.clamp(cos_angles, 0.0, 1.0)  # Ignore back-facing

    # Combine weights
    combined_weights = depth_weights * angle_weights
    weights[valid_mask] = combined_weights

    return weights


def compute_mask_lifting_weights(points_3d, camera, depths, valid_mask):
    weights = compute_visibility_weights(points_3d, camera, depths, valid_mask)
    if valid_mask.any():
        valid_weights = weights[valid_mask]
        if valid_weights.max() <= 1e-8:
            weights[valid_mask] = 1.0
        else:
            weights[valid_mask] = torch.clamp(valid_weights, min=1e-3)
    return weights


def sample_features_at_points(features_2d, points_2d, image_size):
    features = features_2d.permute(2, 0, 1).unsqueeze(0)
    grid = points_2d.unsqueeze(0).unsqueeze(2)

    sampled = F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(2).T


def _sample_mask_region_features(
    points_2d: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_features: torch.Tensor,
    width: int,
    height: int,
):
    if mask_features.shape[0] == 0:
        feature_dim = mask_features.shape[1] if mask_features.dim() == 2 else 0
        return (
            torch.zeros(points_2d.shape[0], feature_dim, device=points_2d.device),
            torch.zeros(points_2d.shape[0], dtype=torch.bool, device=points_2d.device),
            torch.full((points_2d.shape[0],), -1, dtype=torch.long, device=points_2d.device),
        )

    xs = ((points_2d[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5) * (width - 1)
    ys = ((points_2d[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (height - 1)
    xi = torch.clamp(xs.round().long(), 0, width - 1)
    yi = torch.clamp(ys.round().long(), 0, height - 1)

    region_indices = mask_indices[yi, xi]
    has_region = region_indices >= 0
    feature_dim = mask_features.shape[1]

    sampled = torch.zeros(points_2d.shape[0], feature_dim, device=mask_features.device)
    if has_region.any():
        sampled[has_region] = mask_features[region_indices[has_region]]

    return sampled, has_region, region_indices


def _sample_mask_region_ids(
    points_2d: torch.Tensor,
    mask_indices: torch.Tensor,
    width: int,
    height: int,
):
    xs = ((points_2d[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5) * (width - 1)
    ys = ((points_2d[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (height - 1)
    xi = torch.clamp(xs.round().long(), 0, width - 1)
    yi = torch.clamp(ys.round().long(), 0, height - 1)
    region_indices = mask_indices[yi, xi]
    return region_indices, region_indices >= 0


def compute_area_vote_weights(mask_areas, local_mask_ids, area_alpha=0.5, max_weight=4.0):
    if mask_areas is None or mask_areas.numel() == 0 or float(area_alpha) <= 0.0:
        return torch.ones_like(local_mask_ids, dtype=torch.float32)

    valid_area_mask = mask_areas > 0
    if not valid_area_mask.any():
        return torch.ones_like(local_mask_ids, dtype=torch.float32)

    reference_area = torch.median(mask_areas[valid_area_mask].float()).clamp_min(1.0)
    safe_ids = local_mask_ids.clamp(0, mask_areas.shape[0] - 1)
    local_areas = mask_areas[safe_ids].float().clamp_min(1.0)
    weights = torch.pow(reference_area / local_areas, float(area_alpha))
    max_weight = max(1.0, float(max_weight))
    return torch.clamp(weights, min=1.0 / max_weight, max=max_weight)


def _clear_view_device_cache(view_data: MaskFeatureViewData, device) -> None:
    cache = getattr(view_data, "_device_cache", None)
    if isinstance(cache, dict):
        cache.pop(str(device), None)


def _pixel_indices_from_ndc(points_2d: torch.Tensor, width: int, height: int):
    xs = ((points_2d[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5) * (width - 1)
    ys = ((points_2d[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (height - 1)
    xi = torch.clamp(xs.round().long(), 0, width - 1)
    yi = torch.clamp(ys.round().long(), 0, height - 1)
    return xi, yi


def build_projection_depth_buffer(
    points_3d: torch.Tensor,
    camera,
    width: int,
    height: int,
    batch_size: int = 30000,
):
    depth_buffer = torch.full((height * width,), float("inf"), device=points_3d.device)
    n_points = int(points_3d.shape[0])

    for start_idx in range(0, n_points, batch_size):
        end_idx = min(start_idx + batch_size, n_points)
        points_2d, depths, valid = project_points_to_camera(points_3d[start_idx:end_idx], camera)
        if not valid.any():
            continue
        xi, yi = _pixel_indices_from_ndc(points_2d, width, height)
        flat = (yi[valid] * width + xi[valid]).long()
        depth_values = depths[valid].float()
        depth_buffer.scatter_reduce_(0, flat, depth_values, reduce="amin", include_self=True)

    return depth_buffer.view(height, width)


def apply_zbuffer_visibility(
    points_2d: torch.Tensor,
    depths: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_buffer: torch.Tensor,
    width: int,
    height: int,
    abs_tolerance: float = 0.02,
    rel_tolerance: float = 0.03,
):
    if depth_buffer is None:
        return valid_mask

    xi, yi = _pixel_indices_from_ndc(points_2d, width, height)
    nearest_depth = depth_buffer[yi, xi]
    tolerance = float(abs_tolerance) + float(rel_tolerance) * torch.clamp(nearest_depth, min=0.0)
    visible = torch.isfinite(nearest_depth) & (depths <= nearest_depth + tolerance)
    return valid_mask & visible


def lift_dino_features_to_gaussians(
    gaussians_xyz,
    cameras,
    dino_features_2d,
    variance_threshold=0.15,
    min_views=3,
    max_views=100,  
    reduce_dim=None,  
    variance_alpha=3.0,  
    log_hist_bins=10,
):
    
    N = gaussians_xyz.shape[0]
    num_views_total = len(cameras)
    feature_dim = next(iter(dino_features_2d.values())).shape[-1]

    if num_views_total > max_views:
        step = num_views_total // max_views
        sampled_view_indices = list(range(0, num_views_total, step))[:max_views]
        cameras = [cameras[i] for i in sampled_view_indices]
        dino_features_2d = {i: dino_features_2d[sampled_view_indices[i]] for i in range(len(sampled_view_indices))}
        num_views = len(cameras)
    else:
        num_views = num_views_total

    accumulated_features = torch.zeros(N, feature_dim).cuda() 
    accumulated_weights = torch.zeros(N, 1).cuda()  

    feature_sum = torch.zeros(N, feature_dim)  # E[X]
    feature_sq_sum = torch.zeros(N, feature_dim)  # E[X²]
    view_count = torch.zeros(N)  # number of views

    for view_idx, camera in enumerate(cameras):
        # Process in smaller batches to avoid OOM
        batch_size = 30000
        num_batches = (N + batch_size - 1) // batch_size

        sampled_features_full = torch.zeros(N, feature_dim)
        weights_full = torch.zeros(N)  
        valid_mask_full = torch.zeros(N, dtype=torch.bool) 

        features_2d = dino_features_2d[view_idx].cuda()  # Load feature map once per view
        image_size = (camera.image_height, camera.image_width)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_xyz = gaussians_xyz[start_idx:end_idx]

            # 1. Project 3D points to 2D (batch)
            points_2d_batch, depths_batch, valid_mask_batch = project_points_to_camera(batch_xyz, camera)

            # 2. Compute weights (batch)
            weights_batch = compute_visibility_weights(batch_xyz, camera, depths_batch, valid_mask_batch)

            # 3. Sample features from 2D feature map (batch)
            sampled_features_batch = sample_features_at_points(
                features_2d,
                points_2d_batch,
                image_size
            )  # (batch_size, D)

            # Store batch results in CPU tensors
            sampled_features_full[start_idx:end_idx] = sampled_features_batch.cpu()
            weights_full[start_idx:end_idx] = weights_batch.cpu()
            valid_mask_full[start_idx:end_idx] = valid_mask_batch.cpu()

            # Clean up GPU memory after each batch
            del points_2d_batch, depths_batch, valid_mask_batch, weights_batch, sampled_features_batch
            torch.cuda.empty_cache()

        valid_indices = torch.where(valid_mask_full)[0] 
        valid_features = sampled_features_full[valid_indices] 

        feature_sum[valid_indices] += valid_features
        feature_sq_sum[valid_indices] += valid_features ** 2
        view_count[valid_indices] += 1

        weighted_features = sampled_features_full * weights_full.unsqueeze(1)
        accumulated_features += weighted_features.cuda()
        accumulated_weights += weights_full.unsqueeze(1).cuda()  

        del features_2d, sampled_features_full, weights_full, valid_mask_full, weighted_features
        del valid_features, valid_indices
        torch.cuda.empty_cache()

    # 5. Normalize by total weights (already on GPU)
    gaussian_features = accumulated_features / (accumulated_weights + 1e-6)

    if reduce_dim is not None and reduce_dim < feature_dim:
        features_cpu = gaussian_features.cpu()

        # Center the data
        mean_features = features_cpu.mean(dim=0)
        centered_features = features_cpu - mean_features

        # Compute covariance matrix (D x D)
        cov_matrix = torch.mm(centered_features.T, centered_features) / (N - 1)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvectors = eigenvectors[:, idx]

        # Take top reduce_dim components
        pca_components = eigenvectors[:, :reduce_dim]  # (D, reduce_dim)

        # Project features to lower dimension
        reduced_features = torch.mm(centered_features, pca_components)  # (N, reduce_dim)

        # Move back to GPU
        gaussian_features = reduced_features.cuda()
        feature_dim = reduce_dim

        # Variance explained
        variance_explained = eigenvalues[idx[:reduce_dim]].sum() / eigenvalues.sum()

        del features_cpu, mean_features, centered_features, cov_matrix
        del eigenvalues, eigenvectors, pca_components, reduced_features

    feature_mean = feature_sum / (view_count.unsqueeze(1) + 1e-6)  # (N, D)
    feature_var = (feature_sq_sum / (view_count.unsqueeze(1) + 1e-6)) - (feature_mean ** 2)  # (N, D)

    # Average variance across feature dimensions
    feature_variance = feature_var.mean(dim=1)  # (N,)
    feature_variance[view_count < min_views] = float('inf')  # Mark unreliable

    # 7. Reliable mask based on variance and visibility (on CPU)
    reliable_mask = (feature_variance < variance_threshold) & (view_count >= min_views)

    # Soft reliability weighting (higher variance -> lower weight)
    feature_variance_clamped = torch.clamp(feature_variance, min=0.0)
    reliability_weights = torch.exp(-variance_alpha * feature_variance_clamped)
    reliability_weights[view_count < min_views] = 0.0
    reliability_weights = torch.clamp(reliability_weights, 0.0, 1.0)

    feature_variance = feature_variance.cuda()
    reliability_weights = reliability_weights.cuda()
    reliable_mask = reliable_mask.cuda()

    # Statistics
    stats = {
        'total_points': N,
        'reliable_points': reliable_mask.sum().item(),
        'reliability_ratio': (reliable_mask.sum() / N).item(),
        'mean_variance': feature_variance.mean().item(),
        'mean_view_count': view_count.float().mean().item(),
        'mean_weight': reliability_weights.mean().item(),
    }

    return gaussian_features, reliability_weights, stats


def compute_mask_geometric_centers(
    gaussians_xyz,
    cameras,
    feature_field_views,
    view_entries,
    local_to_global_map
):
    device = gaussians_xyz.device
    N = gaussians_xyz.shape[0]

    global_mask_centers = {}  # global_mask_id -> 3D center position

    # For each global mask, collect all 3D points that project to it
    mask_to_points = defaultdict(list)

    for local_idx, (cam_idx, camera, view_data) in enumerate(view_entries):
        tensors_on_device = view_data.to_device(device)
        mask_indices = tensors_on_device["mask_indices"].long()

        if view_data.feature_dim == 0:
            continue

        # Process in batches
        batch_size = 30000
        num_batches = (N + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_xyz = gaussians_xyz[start_idx:end_idx]

            points_2d_batch, depths_batch, valid_mask_batch = project_points_to_camera(batch_xyz, camera)

            # Sample which mask each point belongs to
            xs = ((points_2d_batch[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5) * (view_data.width - 1)
            ys = ((points_2d_batch[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (view_data.height - 1)
            xi = torch.clamp(xs.round().long(), 0, view_data.width - 1)
            yi = torch.clamp(ys.round().long(), 0, view_data.height - 1)

            region_indices = mask_indices[yi, xi]
            has_region = (region_indices >= 0) & valid_mask_batch

            valid_batch_indices = torch.where(has_region)[0]
            for local_idx_in_batch in valid_batch_indices:
                gaussian_idx = (start_idx + local_idx_in_batch).item()
                local_mask_id = region_indices[local_idx_in_batch].item()

                key = (local_idx, local_mask_id)
                global_mask_id = local_to_global_map.get(key, -1)

                if global_mask_id >= 0:
                    mask_to_points[global_mask_id].append(gaussians_xyz[gaussian_idx].cpu())

            del points_2d_batch, depths_batch, valid_mask_batch
            torch.cuda.empty_cache()

    # Compute centers as mean of all points
    for global_mask_id, points in mask_to_points.items():
        if len(points) > 0:
            points_tensor = torch.stack(points)
            global_mask_centers[global_mask_id] = points_tensor.mean(dim=0)

    return global_mask_centers


def build_cross_view_matching_graph(
    all_mask_descriptors,
    similarity_threshold=0.7,
    same_view_threshold=None,
    max_view_gap=None,
    topk_per_mask=3,
    one_to_one_per_view_pair=True,
    mask_geometry_signatures=None,
    geometry_weight=0.35,
    geometry_threshold=0.05,
    geometry_semantic_floor=0.60,
    prevent_view_conflicts=True,
    split_disconnected_geometry=True,
):
    n_masks = len(all_mask_descriptors)

    edges = []
    geometry_bridge_edges = []

    per_mask_edges = defaultdict(list)
    view_pair_edges = defaultdict(list)

    for i, desc_i in enumerate(all_mask_descriptors):
        feat_i = desc_i['feature']
        for j in range(i + 1, n_masks):
            desc_j = all_mask_descriptors[j]

            feat_j = desc_j['feature']

            similarity = torch.nn.functional.cosine_similarity(
                feat_i.unsqueeze(0), feat_j.unsqueeze(0)
            ).item()

            same_view = (desc_i['view_idx'] == desc_j['view_idx'])
            if same_view and same_view_threshold is None:
                continue
            if (
                not same_view
                and max_view_gap is not None
                and max_view_gap > 0
                and abs(desc_i['view_idx'] - desc_j['view_idx']) > max_view_gap
            ):
                continue

            threshold = same_view_threshold if same_view else similarity_threshold
            geometry_overlap = 0.0
            has_geometry = False
            if mask_geometry_signatures and not same_view:
                key_i = (desc_i["view_idx"], desc_i["local_mask_id"])
                key_j = (desc_j["view_idx"], desc_j["local_mask_id"])
                sig_i = mask_geometry_signatures.get(key_i)
                sig_j = mask_geometry_signatures.get(key_j)
                if sig_i is not None and sig_j is not None and len(sig_i) > 0 and len(sig_j) > 0:
                    has_geometry = True
                    if len(sig_i) < len(sig_j):
                        intersection = sum(1 for item in sig_i if item in sig_j)
                    else:
                        intersection = sum(1 for item in sig_j if item in sig_i)
                    geometry_overlap = intersection / max(1, min(len(sig_i), len(sig_j)))

            if has_geometry and not same_view:
                combined_score = (
                    (1.0 - geometry_weight) * similarity
                    + geometry_weight * geometry_overlap
                )
                accept = (
                    (similarity > threshold and geometry_overlap > 0.0)
                    or (
                        geometry_overlap >= geometry_threshold
                        and similarity >= geometry_semantic_floor
                    )
                )
                edge_score = combined_score
            else:
                accept = similarity > threshold
                edge_score = similarity

            # Apply threshold
            if accept:
                edge = (i, j, edge_score, similarity, geometry_overlap)
                if (
                    has_geometry
                    and not same_view
                    and geometry_overlap >= geometry_threshold
                    and similarity >= geometry_semantic_floor
                ):
                    # Keep strong 3D co-visibility links even if top-k / view-pair
                    # pruning would otherwise discard them. These bridge edges are
                    # what let short per-view tracks form long object-consistent IDs.
                    geometry_bridge_edges.append(edge)
                if one_to_one_per_view_pair and not same_view:
                    view_key = (min(desc_i['view_idx'], desc_j['view_idx']), max(desc_i['view_idx'], desc_j['view_idx']))
                    view_pair_edges[view_key].append(edge)
                else:
                    edges.append(edge)

    if one_to_one_per_view_pair:
        for _, candidates in view_pair_edges.items():
            used_i = set()
            used_j = set()
            for i, j, score, similarity, geometry_overlap in sorted(candidates, key=lambda x: x[2], reverse=True):
                if i in used_i or j in used_j:
                    continue
                edges.append((i, j, score, similarity, geometry_overlap))
                used_i.add(i)
                used_j.add(j)

    if topk_per_mask is not None and topk_per_mask > 0:
        for edge in edges:
            i, j = edge[0], edge[1]
            per_mask_edges[i].append(edge)
            per_mask_edges[j].append(edge)
        keep = set()
        for mask_edges in per_mask_edges.values():
            for edge in sorted(mask_edges, key=lambda x: x[2], reverse=True)[:topk_per_mask]:
                keep.add(edge)
        edges = list(keep)

    if geometry_bridge_edges:
        edge_set = set(edges)
        for edge in geometry_bridge_edges:
            if edge not in edge_set:
                edges.append(edge)
                edge_set.add(edge)

    parent = list(range(n_masks))
    cluster_views = [{all_mask_descriptors[i]["view_idx"]} for i in range(n_masks)]

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            if prevent_view_conflicts and cluster_views[px].intersection(cluster_views[py]):
                return False
            parent[px] = py
            cluster_views[py].update(cluster_views[px])
            return True
        return False

    # Sort edges by similarity (descending) for greedy matching
    edges.sort(key=lambda x: x[2], reverse=True)

    accepted_edges = 0
    skipped_conflicts = 0
    for i, j, score, similarity, geometry_overlap in edges:
        if union(i, j):
            accepted_edges += 1
        else:
            skipped_conflicts += 1

    # Extract clusters
    clusters = defaultdict(list)
    for i in range(n_masks):
        root = find(i)
        clusters[root].append(i)

    def _geometry_overlap_for_indices(idx_a, idx_b):
        desc_a = all_mask_descriptors[idx_a]
        desc_b = all_mask_descriptors[idx_b]
        if desc_a["view_idx"] == desc_b["view_idx"]:
            return 0.0
        key_a = (desc_a["view_idx"], desc_a["local_mask_id"])
        key_b = (desc_b["view_idx"], desc_b["local_mask_id"])
        sig_a = mask_geometry_signatures.get(key_a) if mask_geometry_signatures else None
        sig_b = mask_geometry_signatures.get(key_b) if mask_geometry_signatures else None
        if sig_a is None or sig_b is None or len(sig_a) == 0 or len(sig_b) == 0:
            return 0.0
        if len(sig_a) < len(sig_b):
            intersection = sum(1 for item in sig_a if item in sig_b)
        else:
            intersection = sum(1 for item in sig_b if item in sig_a)
        return intersection / max(1, min(len(sig_a), len(sig_b)))

    def _semantic_similarity_for_indices(idx_a, idx_b):
        feat_a = all_mask_descriptors[idx_a]["feature"]
        feat_b = all_mask_descriptors[idx_b]["feature"]
        return torch.nn.functional.cosine_similarity(
            feat_a.unsqueeze(0), feat_b.unsqueeze(0)
        ).item()

    def _split_cluster_by_geometry(cluster_members):
        if (
            not split_disconnected_geometry
            or not mask_geometry_signatures
            or len(cluster_members) <= 2
        ):
            return [cluster_members]

        local_parent = {idx: idx for idx in cluster_members}
        local_views = {idx: {all_mask_descriptors[idx]["view_idx"]} for idx in cluster_members}

        def local_find(x):
            if local_parent[x] != x:
                local_parent[x] = local_find(local_parent[x])
            return local_parent[x]

        def local_union(a, b):
            ra, rb = local_find(a), local_find(b)
            if ra == rb:
                return
            if prevent_view_conflicts and local_views[ra].intersection(local_views[rb]):
                return
            local_parent[ra] = rb
            local_views[rb].update(local_views[ra])

        high_semantic_threshold = max(0.90, similarity_threshold + 0.15)
        for offset, idx_a in enumerate(cluster_members):
            desc_a = all_mask_descriptors[idx_a]
            key_a = (desc_a["view_idx"], desc_a["local_mask_id"])
            has_sig_a = key_a in mask_geometry_signatures
            for idx_b in cluster_members[offset + 1:]:
                desc_b = all_mask_descriptors[idx_b]
                if desc_a["view_idx"] == desc_b["view_idx"]:
                    continue

                key_b = (desc_b["view_idx"], desc_b["local_mask_id"])
                has_sig_b = key_b in mask_geometry_signatures
                geometry_overlap = _geometry_overlap_for_indices(idx_a, idx_b)
                if geometry_overlap >= geometry_threshold:
                    local_union(idx_a, idx_b)
                    continue

                if not (has_sig_a and has_sig_b):
                    similarity = _semantic_similarity_for_indices(idx_a, idx_b)
                    if similarity >= high_semantic_threshold:
                        local_union(idx_a, idx_b)

        split_groups = defaultdict(list)
        for idx in cluster_members:
            split_groups[local_find(idx)].append(idx)
        return sorted(split_groups.values(), key=lambda members: min(members))

    # Assign global IDs
    local_to_global_map = {}
    global_mask_id = 0

    ordered_clusters = sorted(clusters.values(), key=lambda members: min(members))
    split_cluster_count = 0
    refined_clusters = []
    for cluster_members_indices in ordered_clusters:
        split_clusters = _split_cluster_by_geometry(cluster_members_indices)
        if len(split_clusters) > 1:
            split_cluster_count += 1
        refined_clusters.extend(split_clusters)

    ordered_clusters = sorted(refined_clusters, key=lambda members: min(members))
    for cluster_members_indices in ordered_clusters:
        for idx in cluster_members_indices:
            desc = all_mask_descriptors[idx]
            key = (desc['view_idx'], desc['local_mask_id'])
            local_to_global_map[key] = global_mask_id
        global_mask_id += 1

    return local_to_global_map, global_mask_id


def build_mask_geometry_signatures(
    gaussians_xyz,
    view_entries,
    max_sample_points=120000,
    max_points_per_mask=2048,
    min_points_per_mask=8,
    use_zbuffer_visibility=True,
    zbuffer_abs_tolerance=0.02,
    zbuffer_rel_tolerance=0.03,
):
    device = gaussians_xyz.device
    n_points = int(gaussians_xyz.shape[0])
    if n_points == 0 or not view_entries:
        return {}

    sample_count = min(n_points, int(max_sample_points))
    if sample_count <= 0:
        return {}

    if sample_count == n_points:
        sample_indices = torch.arange(n_points, device=device)
    else:
        sample_indices = torch.linspace(
            0,
            n_points - 1,
            steps=sample_count,
            device=device,
        ).long()
    sample_xyz = gaussians_xyz[sample_indices]
    sample_indices_cpu = sample_indices.detach().cpu().numpy()

    signatures = {}

    batch_size = 30000
    for view_idx, (cam_idx, camera, view_data) in enumerate(view_entries):
        tensors_on_device = view_data.to_device(device)
        mask_indices = tensors_on_device["mask_indices"].long()
        num_masks = int(tensors_on_device["features"].shape[0])
        per_mask = [set() for _ in range(num_masks)]
        depth_buffer = None
        if use_zbuffer_visibility:
            depth_buffer = build_projection_depth_buffer(
                sample_xyz,
                camera,
                view_data.width,
                view_data.height,
                batch_size=batch_size,
            )

        for start_idx in range(0, sample_count, batch_size):
            end_idx = min(start_idx + batch_size, sample_count)
            points_2d, depths, valid_mask = project_points_to_camera(sample_xyz[start_idx:end_idx], camera)
            valid_mask = apply_zbuffer_visibility(
                points_2d,
                depths,
                valid_mask,
                depth_buffer,
                view_data.width,
                view_data.height,
                abs_tolerance=zbuffer_abs_tolerance,
                rel_tolerance=zbuffer_rel_tolerance,
            )
            region_ids, has_region = _sample_mask_region_ids(
                points_2d,
                mask_indices,
                view_data.width,
                view_data.height,
            )
            valid = (valid_mask & has_region).detach().cpu().numpy()
            if not valid.any():
                continue

            region_ids_cpu = region_ids.detach().cpu().numpy()
            batch_sample_ids = sample_indices_cpu[start_idx:end_idx]
            for local_mask_id in np.unique(region_ids_cpu[valid]):
                if local_mask_id < 0 or local_mask_id >= num_masks:
                    continue
                current = per_mask[int(local_mask_id)]
                if len(current) >= max_points_per_mask:
                    continue
                ids = batch_sample_ids[valid & (region_ids_cpu == local_mask_id)]
                remaining = max_points_per_mask - len(current)
                if ids.shape[0] > remaining:
                    stride = max(1, int(np.ceil(ids.shape[0] / remaining)))
                    ids = ids[::stride][:remaining]
                current.update(int(x) for x in ids)

            del points_2d, depths, valid_mask, region_ids, has_region
            torch.cuda.empty_cache()

        for local_mask_id, signature in enumerate(per_mask):
            if len(signature) >= min_points_per_mask:
                signatures[(view_idx, local_mask_id)] = frozenset(signature)

        del mask_indices, depth_buffer
        _clear_view_device_cache(view_data, device)
        torch.cuda.empty_cache()

    return signatures


def refine_matching_with_multiview_consensus(
    local_to_global_map,
    all_mask_descriptors,
    view_entries,
    min_consensus_views=2
):
    # Group masks by global ID
    global_to_locals = defaultdict(list)
    for (view_idx, local_mask_id), global_id in local_to_global_map.items():
        global_to_locals[global_id].append((view_idx, local_mask_id))

    # Check how many different views each global object appears in
    refined_mapping = {}
    new_global_id = 0

    for global_id, local_masks in global_to_locals.items():
        # Count unique views
        unique_views = len(set(view_idx for view_idx, _ in local_masks))

        # If appears in enough views, keep the global ID
        if unique_views >= min_consensus_views or len(local_masks) >= 3:
            for view_idx, local_mask_id in local_masks:
                refined_mapping[(view_idx, local_mask_id)] = new_global_id
            new_global_id += 1
        else:
            # Assign separate IDs (treat as different objects)
            for view_idx, local_mask_id in local_masks:
                refined_mapping[(view_idx, local_mask_id)] = new_global_id
                new_global_id += 1

    return refined_mapping, new_global_id


def lift_feature_field_masks_to_gaussians(
    gaussians_xyz,
    cameras,
    feature_field_views: Dict[str, MaskFeatureViewData],
    variance_threshold=0.05,
    min_views=2,
    variance_alpha=2.0,
    log_hist_bins=10,
    use_improved_matching=True,
    min_consensus_views=2,
    matching_threshold=0.7,
    same_view_matching_threshold=None,
    use_multiview_refinement=False,
    filter_mask_ids_by_reliability=False,
    matching_max_view_gap=10,
    matching_topk_per_mask=3,
    matching_one_to_one=True,
    use_geometry_matching=True,
    matching_geometry_sample_points=120000,
    matching_geometry_max_points_per_mask=2048,
    matching_geometry_weight=0.35,
    matching_geometry_threshold=0.05,
    matching_geometry_semantic_floor=0.60,
    prevent_view_conflicts=True,
    use_zbuffer_visibility=True,
    zbuffer_abs_tolerance=0.02,
    zbuffer_rel_tolerance=0.03,
    mask_vote_area_alpha=0.5,
    mask_vote_max_weight=4.0,
    mask_vote_min_count=1,
    mask_vote_min_score_ratio=0.0,
    mask_vote_min_margin=0.0,
):
    device = gaussians_xyz.device
    N = gaussians_xyz.shape[0]

    view_entries = []
    for idx, camera in enumerate(cameras):
        view_data = feature_field_views.get(camera.image_name)
        if view_data is None or view_data.feature_dim == 0:
            continue
        view_entries.append((idx, camera, view_data))

    if not view_entries:
        raise ValueError("No matching feature-field views were found for the provided cameras.")

    feature_dim = view_entries[0][2].feature_dim
    mask_geometry_signatures = None
    if use_geometry_matching:
        mask_geometry_signatures = build_mask_geometry_signatures(
            gaussians_xyz,
            view_entries,
            max_sample_points=matching_geometry_sample_points,
            max_points_per_mask=matching_geometry_max_points_per_mask,
            use_zbuffer_visibility=use_zbuffer_visibility,
            zbuffer_abs_tolerance=zbuffer_abs_tolerance,
            zbuffer_rel_tolerance=zbuffer_rel_tolerance,
        )

    # Collect all mask features from all views
    all_mask_descriptors = []  # List of (view_idx, local_mask_id, feature_vector)
    for view_idx, (cam_idx, camera, view_data) in enumerate(view_entries):
        tensors = view_data.to_device('cpu')
        mask_features = tensors["features"]  # (num_masks, feature_dim)
        for local_mask_id in range(mask_features.shape[0]):
            all_mask_descriptors.append({
                'view_idx': view_idx,
                'cam_idx': cam_idx,
                'local_mask_id': local_mask_id,
                'feature': mask_features[local_mask_id].cpu()
            })

    similarity_threshold = matching_threshold
    same_view_threshold = same_view_matching_threshold

    if use_improved_matching:
        # Step 1: Build matching graph with Union-Find
        local_to_global_map, global_mask_id = build_cross_view_matching_graph(
            all_mask_descriptors,
            similarity_threshold=similarity_threshold,
            same_view_threshold=same_view_threshold,
            max_view_gap=matching_max_view_gap,
            topk_per_mask=matching_topk_per_mask,
            one_to_one_per_view_pair=matching_one_to_one,
            mask_geometry_signatures=mask_geometry_signatures,
            geometry_weight=matching_geometry_weight,
            geometry_threshold=matching_geometry_threshold,
            geometry_semantic_floor=matching_geometry_semantic_floor,
            prevent_view_conflicts=prevent_view_conflicts,
        )

        if use_multiview_refinement:
            local_to_global_map, global_mask_id = refine_matching_with_multiview_consensus(
                local_to_global_map,
                all_mask_descriptors,
                view_entries,
                min_consensus_views=min_consensus_views
            )
        else:
            print(f"Skip multi-view refinement")
    else:
        # Original greedy matching (fallback)
        local_to_global_map = {}  # (view_idx, local_mask_id) -> global_mask_id
        global_mask_id = 0
        assigned = set()

        for i, desc_i in enumerate(all_mask_descriptors):
            key_i = (desc_i['view_idx'], desc_i['local_mask_id'])
            if key_i in assigned:
                continue

            # Start new cluster
            cluster_members = [key_i]
            assigned.add(key_i)

            # Find all similar masks (including from SAME view for part merging)
            feat_i = desc_i['feature']
            for j, desc_j in enumerate(all_mask_descriptors):
                if i == j:
                    continue
                key_j = (desc_j['view_idx'], desc_j['local_mask_id'])
                if key_j in assigned:
                    continue

                # Compute cosine similarity
                feat_j = desc_j['feature']
                similarity = torch.nn.functional.cosine_similarity(
                    feat_i.unsqueeze(0), feat_j.unsqueeze(0)
                ).item()

                same_view = (desc_i['view_idx'] == desc_j['view_idx'])
                if same_view and same_view_threshold is None:
                    continue
                threshold = same_view_threshold if same_view else similarity_threshold

                if similarity > threshold:
                    cluster_members.append(key_j)
                    assigned.add(key_j)

            # Assign global ID to all members of this cluster
            for member_key in cluster_members:
                local_to_global_map[member_key] = global_mask_id

            global_mask_id += 1

    global_mask_count = int(global_mask_id)

    accumulated_features = torch.zeros(N, feature_dim, device=device)
    accumulated_weights = torch.zeros(N, 1, device=device)
    view_count = torch.zeros(N)

    mask_id_votes = {}  # gaussian_idx -> {global_mask_id: weighted score}
    mask_id_vote_counts = {}  # gaussian_idx -> {global_mask_id: raw vote count}

    for local_idx, (cam_idx, camera, view_data) in enumerate(view_entries):
        tensors_on_device = view_data.to_device(device)
        mask_indices = tensors_on_device["mask_indices"].long()
        mask_features = tensors_on_device["features"]
        mask_areas = tensors_on_device.get("areas", None)

        if mask_features.shape[0] == 0:
            continue

        batch_size = 30000
        num_batches = (N + batch_size - 1) // batch_size
        depth_buffer = None
        if use_zbuffer_visibility:
            depth_buffer = build_projection_depth_buffer(
                gaussians_xyz,
                camera,
                view_data.width,
                view_data.height,
                batch_size=batch_size,
            )

        sampled_features_full = torch.zeros(N, feature_dim)
        weights_full = torch.zeros(N)
        valid_mask_full = torch.zeros(N, dtype=torch.bool)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_xyz = gaussians_xyz[start_idx:end_idx]

            points_2d_batch, depths_batch, valid_mask_batch = project_points_to_camera(batch_xyz, camera)
            valid_mask_batch = apply_zbuffer_visibility(
                points_2d_batch,
                depths_batch,
                valid_mask_batch,
                depth_buffer,
                view_data.width,
                view_data.height,
                abs_tolerance=zbuffer_abs_tolerance,
                rel_tolerance=zbuffer_rel_tolerance,
            )

            sampled_features_batch, has_region, region_ids_batch = _sample_mask_region_features(
                points_2d_batch, mask_indices, mask_features, view_data.width, view_data.height
            )
            valid_mask_batch = valid_mask_batch & has_region

            valid_batch_indices = torch.where(valid_mask_batch)[0]
            valid_region_ids = region_ids_batch[valid_batch_indices].long()
            area_vote_weights = compute_area_vote_weights(
                mask_areas,
                valid_region_ids,
                area_alpha=float(mask_vote_area_alpha),
                max_weight=float(mask_vote_max_weight),
            )
            for vote_offset, local_idx_in_batch in enumerate(valid_batch_indices):
                gaussian_idx = (start_idx + local_idx_in_batch).item()
                local_mask_id = region_ids_batch[local_idx_in_batch].item()

                # Convert local mask ID to global mask ID
                view_idx = local_idx  # This is the view index in view_entries
                key = (view_idx, local_mask_id)
                mapped_global_mask_id = local_to_global_map.get(key, -1)

                # Skip if no global mapping found
                if mapped_global_mask_id < 0:
                    continue

                if gaussian_idx not in mask_id_votes:
                    mask_id_votes[gaussian_idx] = {}
                    mask_id_vote_counts[gaussian_idx] = {}
                if mapped_global_mask_id not in mask_id_votes[gaussian_idx]:
                    mask_id_votes[gaussian_idx][mapped_global_mask_id] = 0.0
                    mask_id_vote_counts[gaussian_idx][mapped_global_mask_id] = 0
                mask_id_votes[gaussian_idx][mapped_global_mask_id] += float(area_vote_weights[vote_offset].item())
                mask_id_vote_counts[gaussian_idx][mapped_global_mask_id] += 1

            if valid_mask_batch.any():
                weights_batch = compute_mask_lifting_weights(batch_xyz, camera, depths_batch, valid_mask_batch)
            else:
                weights_batch = torch.zeros_like(valid_mask_batch, dtype=torch.float32, device=device)

            sampled_features_full[start_idx:end_idx] = sampled_features_batch.detach().cpu()
            weights_full[start_idx:end_idx] = weights_batch.detach().cpu()
            valid_mask_full[start_idx:end_idx] = valid_mask_batch.detach().cpu()

            del points_2d_batch, depths_batch, valid_mask_batch, sampled_features_batch, has_region, weights_batch
            torch.cuda.empty_cache()

        valid_indices = torch.where(valid_mask_full)[0]
        if valid_indices.numel() == 0:
            continue

        view_count[valid_indices] += 1

        weighted_features = sampled_features_full * weights_full.unsqueeze(1)
        accumulated_features += weighted_features.to(device)
        accumulated_weights += weights_full.unsqueeze(1).to(device)

        del sampled_features_full, weights_full, valid_mask_full, valid_indices, weighted_features, depth_buffer
        _clear_view_device_cache(view_data, device)
        torch.cuda.empty_cache()

    gaussian_features = accumulated_features / (accumulated_weights + 1e-6)
    gaussian_mask_ids = torch.full((N,), -1, dtype=torch.long)

    rejected_by_confidence = 0
    for gaussian_idx, votes in mask_id_votes.items():
        if votes:
            sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
            best_mask_id, best_score = sorted_votes[0]
            second_score = sorted_votes[1][1] if len(sorted_votes) > 1 else 0.0
            total_score = sum(votes.values())
            raw_count = mask_id_vote_counts.get(gaussian_idx, {}).get(best_mask_id, 0)

            score_ratio = best_score / max(total_score, 1e-6)
            score_margin = (best_score - second_score) / max(total_score, 1e-6)
            if (
                raw_count >= int(mask_vote_min_count)
                and score_ratio >= float(mask_vote_min_score_ratio)
                and score_margin >= float(mask_vote_min_margin)
            ):
                gaussian_mask_ids[gaussian_idx] = best_mask_id
            else:
                rejected_by_confidence += 1

    voted_mask_ids = gaussian_mask_ids.clone()
    num_voted = (voted_mask_ids >= 0).sum().item()
    num_voted_unique = len(torch.unique(voted_mask_ids[voted_mask_ids >= 0]))
    voted_id_counts = {}
    if num_voted_unique > 0:
        voted_ids, voted_counts = torch.unique(voted_mask_ids[voted_mask_ids >= 0], return_counts=True)
        top_order = torch.argsort(voted_counts, descending=True)[:20]
        top_summary = []
        for idx in top_order:
            mask_id = int(voted_ids[idx].item())
            count = int(voted_counts[idx].item())
            voted_id_counts[str(mask_id)] = count
            top_summary.append((mask_id, count))
        for mask_id, count in zip(voted_ids.cpu().numpy().tolist(), voted_counts.cpu().numpy().tolist()):
            voted_id_counts[str(int(mask_id))] = int(count)

    variance_sum = torch.zeros(N)

    for local_idx, (cam_idx, camera, view_data) in enumerate(view_entries):
        tensors_on_device = view_data.to_device(device)
        mask_indices = tensors_on_device["mask_indices"].long()
        mask_features = tensors_on_device["features"]

        if mask_features.shape[0] == 0:
            continue

        batch_size = 30000
        num_batches = (N + batch_size - 1) // batch_size
        depth_buffer = None
        if use_zbuffer_visibility:
            depth_buffer = build_projection_depth_buffer(
                gaussians_xyz,
                camera,
                view_data.width,
                view_data.height,
                batch_size=batch_size,
            )

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_xyz = gaussians_xyz[start_idx:end_idx]

            points_2d_batch, depths_batch, valid_mask_batch = project_points_to_camera(batch_xyz, camera)
            valid_mask_batch = apply_zbuffer_visibility(
                points_2d_batch,
                depths_batch,
                valid_mask_batch,
                depth_buffer,
                view_data.width,
                view_data.height,
                abs_tolerance=zbuffer_abs_tolerance,
                rel_tolerance=zbuffer_rel_tolerance,
            )
            sampled_features_batch, has_region, _ = _sample_mask_region_features(
                points_2d_batch, mask_indices, mask_features, view_data.width, view_data.height
            )
            valid_mask_batch = valid_mask_batch & has_region

            residual_batch = ((sampled_features_batch - gaussian_features[start_idx:end_idx]) ** 2).mean(dim=1)
            variance_sum[start_idx:end_idx] += (residual_batch * valid_mask_batch.float()).detach().cpu()

            del points_2d_batch, depths_batch, valid_mask_batch, sampled_features_batch, has_region, residual_batch
            torch.cuda.empty_cache()

        del depth_buffer
        _clear_view_device_cache(view_data, device)

    feature_variance = variance_sum / (view_count + 1e-6)
    feature_variance[view_count < min_views] = float("inf")

    reliable_mask = (feature_variance < variance_threshold) & (view_count >= min_views)
    feature_variance_clamped = torch.clamp(feature_variance, min=0.0)
    reliability_weights = torch.exp(-variance_alpha * feature_variance_clamped)
    reliability_weights[~reliable_mask] = 0.0
    reliability_weights = torch.clamp(reliability_weights, 0.0, 1.0)

    if filter_mask_ids_by_reliability:
        gaussian_mask_ids[~reliable_mask] = -1
    else:
        # Keep geometric/mask identity voting independent from feature supervision reliability.
        # Reliability still controls semantic feature losses, but low feature confidence should
        # not erase IDs that are needed for qualitative object visualization/editing.
        gaussian_mask_ids = voted_mask_ids
        gaussian_mask_ids[view_count < min_views] = -1

    feature_variance = feature_variance.cuda()
    reliability_weights = reliability_weights.cuda()
    reliable_mask = reliable_mask.cuda()

    finite_variance = feature_variance[torch.isfinite(feature_variance)]
    mean_variance = finite_variance.mean().item() if finite_variance.numel() > 0 else float("inf")

    stats = {
        "total_points": N,
        "reliable_points": reliable_mask.sum().item(),
        "reliability_ratio": (reliable_mask.sum() / N).item(),
        "mean_variance": mean_variance,
        "mean_view_count": view_count.float().mean().item(),
        "mean_weight": reliability_weights.mean().item(),
        "voted_mask_points": num_voted,
        "voted_unique_masks": num_voted_unique,
        "mask_ids_filter_by_reliability": filter_mask_ids_by_reliability,
        "view_image_names": [camera.image_name for _, camera, _ in view_entries],
        "local_to_global_map": [
            {
                "view_idx": int(view_idx),
                "local_mask_id": int(local_mask_id),
                "global_mask_id": int(global_id),
            }
            for (view_idx, local_mask_id), global_id in sorted(local_to_global_map.items())
        ],
        "global_mask_count": int(global_mask_count),
        "matching_threshold": float(matching_threshold),
        "same_view_matching_threshold": (
            None if same_view_matching_threshold is None else float(same_view_matching_threshold)
        ),
        "matching_max_view_gap": (
            None if matching_max_view_gap is None else int(matching_max_view_gap)
        ),
        "matching_topk_per_mask": (
            None if matching_topk_per_mask is None else int(matching_topk_per_mask)
        ),
        "matching_one_to_one": bool(matching_one_to_one),
        "use_geometry_matching": bool(use_geometry_matching),
        "matching_geometry_sample_points": int(matching_geometry_sample_points),
        "matching_geometry_max_points_per_mask": int(matching_geometry_max_points_per_mask),
        "matching_geometry_weight": float(matching_geometry_weight),
        "matching_geometry_threshold": float(matching_geometry_threshold),
        "matching_geometry_semantic_floor": float(matching_geometry_semantic_floor),
        "prevent_view_conflicts": bool(prevent_view_conflicts),
        "use_zbuffer_visibility": bool(use_zbuffer_visibility),
        "zbuffer_abs_tolerance": float(zbuffer_abs_tolerance),
        "zbuffer_rel_tolerance": float(zbuffer_rel_tolerance),
        "geometry_signature_masks": (
            0 if mask_geometry_signatures is None else int(len(mask_geometry_signatures))
        ),
        "mask_vote_area_alpha": float(mask_vote_area_alpha),
        "mask_vote_max_weight": float(mask_vote_max_weight),
        "mask_vote_min_count": int(mask_vote_min_count),
        "mask_vote_min_score_ratio": float(mask_vote_min_score_ratio),
        "mask_vote_min_margin": float(mask_vote_min_margin),
        "voted_id_counts": voted_id_counts,
    }
    return gaussian_features, reliability_weights, stats, gaussian_mask_ids

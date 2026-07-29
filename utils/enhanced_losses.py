import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def multiscale_semantic_cohesion_loss(
    gaussians_xyz: torch.Tensor,
    identity_encoding: torch.Tensor,
    dino_features_fine: torch.Tensor,
    dino_features_coarse: Optional[torch.Tensor] = None,
    k: int = 10,
    spatial_threshold: float = 0.5,
    lambda_val: float = 1.0,
    fine_weight: float = 0.6,
    coarse_weight: float = 0.4
) -> torch.Tensor:
    
    # Fine-scale loss (precise boundaries)
    loss_fine = _semantic_cohesion_single_scale(
        gaussians_xyz,
        identity_encoding,
        dino_features_fine,
        k=k,
        spatial_threshold=spatial_threshold,
        semantic_threshold=0.75 
    )

    if dino_features_coarse is None:
        return lambda_val * loss_fine

    # Coarse-scale loss (semantic consistency)
    loss_coarse = _semantic_cohesion_single_scale(
        gaussians_xyz,
        identity_encoding,
        dino_features_coarse,
        k=k,
        spatial_threshold=spatial_threshold * 1.5,
        semantic_threshold=0.65 
    )

    # Weighted combination
    total_loss = fine_weight * loss_fine + coarse_weight * loss_coarse
    return lambda_val * total_loss


def _semantic_cohesion_single_scale(
    gaussians_xyz: torch.Tensor,
    identity_encoding: torch.Tensor,
    dino_features: torch.Tensor,
    k: int,
    spatial_threshold: float,
    semantic_threshold: float
) -> torch.Tensor:
    N = gaussians_xyz.shape[0]

    # Memory optimization: sample subset
    max_samples = 5000
    if N > max_samples:
        sample_indices = torch.randperm(N, device='cuda')[:max_samples]
        sample_xyz = gaussians_xyz[sample_indices]
        sample_identity = identity_encoding[sample_indices]
        sample_dino = dino_features[sample_indices]
    else:
        sample_xyz = gaussians_xyz
        sample_identity = identity_encoding
        sample_dino = dino_features

    N_sample = sample_xyz.shape[0]

    # Find k-NN in 3D space (batched)
    batch_size = 1000
    topk_indices_list = []
    topk_dists_list = []

    for i in range(0, N_sample, batch_size):
        end_i = min(i + batch_size, N_sample)
        batch_xyz = sample_xyz[i:end_i]

        dists_batch = torch.cdist(batch_xyz, gaussians_xyz)
        topk_dists_batch, topk_indices_batch = dists_batch.topk(k+1, largest=False, dim=1)

        topk_indices_list.append(topk_indices_batch[:, 1:])
        topk_dists_list.append(topk_dists_batch[:, 1:])

        del dists_batch, topk_dists_batch, topk_indices_batch
        torch.cuda.empty_cache()

    topk_indices = torch.cat(topk_indices_list, dim=0)
    topk_dists = torch.cat(topk_dists_list, dim=0)

    spatial_mask = topk_dists < spatial_threshold

    # DINO feature similarity
    sample_dino_norm = F.normalize(sample_dino, p=2, dim=1)
    all_dino_norm = F.normalize(dino_features, p=2, dim=1)

    neighbor_dino = all_dino_norm[topk_indices]
    dino_similarity = torch.sum(
        sample_dino_norm.unsqueeze(1) * neighbor_dino,
        dim=2
    )

    semantic_mask = dino_similarity > semantic_threshold
    valid_mask = spatial_mask & semantic_mask

    if valid_mask.sum() == 0:
        return torch.tensor(0.0).cuda()

    # Identity similarity
    sample_identity_norm = F.normalize(sample_identity, p=2, dim=1)
    all_identity_norm = F.normalize(identity_encoding, p=2, dim=1)

    neighbor_identity = all_identity_norm[topk_indices]
    identity_similarity = torch.sum(
        sample_identity_norm.unsqueeze(1) * neighbor_identity,
        dim=2
    )

    # Loss: match identity similarity to DINO similarity
    loss = F.mse_loss(
        identity_similarity[valid_mask],
        dino_similarity[valid_mask]
    )

    return loss


def geometric_constraint_loss(
    gaussians_xyz: torch.Tensor,
    gaussians_scale: torch.Tensor,
    identity_encoding: torch.Tensor,
    predicted_ids: torch.Tensor,
    k: int = 10,
    lambda_val: float = 1.0
) -> torch.Tensor:

    N = gaussians_xyz.shape[0]

    # Sample for efficiency
    max_samples = 3000
    if N > max_samples:
        sample_indices = torch.randperm(N, device='cuda')[:max_samples]
        sample_xyz = gaussians_xyz[sample_indices]
        sample_scale = gaussians_scale[sample_indices]
        sample_ids = predicted_ids[sample_indices]
    else:
        sample_xyz = gaussians_xyz
        sample_scale = gaussians_scale
        sample_ids = predicted_ids

    N_sample = sample_xyz.shape[0]

    # Find k-NN
    dists = torch.cdist(sample_xyz, gaussians_xyz)
    _, neighbor_indices = dists.topk(k+1, largest=False, dim=1)
    neighbor_indices = neighbor_indices[:, 1:]  # Exclude self

    # Get neighbor properties
    neighbor_xyz = gaussians_xyz[neighbor_indices]  # (N_sample, k, 3)
    neighbor_scale = gaussians_scale[neighbor_indices]  # (N_sample, k, 3)
    neighbor_ids = predicted_ids[neighbor_indices]  # (N_sample, k)

    # 1. Surface normal consistency
    sample_normals = []
    neighbor_normals = []

    for i in range(N_sample):
        # Sample point normal
        local_points = neighbor_xyz[i]  # (k, 3)
        centered = local_points - sample_xyz[i:i+1]

        cov = torch.mm(centered.T, centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue

        sample_normals.append(normal)

        for j in range(k):
            nb_idx = neighbor_indices[i, j]
            nb_local_indices = neighbor_indices[i]  # Use same neighborhood
            nb_local_points = gaussians_xyz[nb_local_indices]
            nb_centered = nb_local_points - gaussians_xyz[nb_idx:nb_idx+1]

            nb_cov = torch.mm(nb_centered.T, nb_centered)
            nb_eigenvalues, nb_eigenvectors = torch.linalg.eigh(nb_cov)
            nb_normal = nb_eigenvectors[:, 0]

            neighbor_normals.append(nb_normal)

    if len(sample_normals) == 0:
        return torch.tensor(0.0).cuda()

    sample_normals = torch.stack(sample_normals)  # (N_sample, 3)
    neighbor_normals = torch.stack(neighbor_normals).view(N_sample, k, 3)

    # Normal consistency loss (only for same-ID neighbors)
    same_id_mask = (sample_ids.unsqueeze(1) == neighbor_ids)  # (N_sample, k)

    if same_id_mask.sum() == 0:
        normal_loss = torch.tensor(0.0).cuda()
    else:
        # Cosine similarity between normals
        normal_sim = torch.sum(
            sample_normals.unsqueeze(1) * neighbor_normals,
            dim=2
        )  # (N_sample, k)

        # Loss: encourage alignment (normal_sim close to 1 or -1)
        # Use 1 - |normal_sim| so that 0 = aligned, 1 = perpendicular
        normal_consistency = 1.0 - torch.abs(normal_sim)
        normal_loss = normal_consistency[same_id_mask].mean()

    # 2. Scale consistency
    # Same-ID neighbors should have similar scales
    scale_diff = torch.abs(
        sample_scale.unsqueeze(1) - neighbor_scale
    )  # (N_sample, k, 3)

    scale_loss_per_nb = scale_diff.mean(dim=2)  # (N_sample, k)

    if same_id_mask.sum() == 0:
        scale_loss = torch.tensor(0.0).cuda()
    else:
        scale_loss = scale_loss_per_nb[same_id_mask].mean()

    # 3. Identity encoding smoothness
    # Same-ID neighbors should have similar identity encodings
    sample_identity = identity_encoding[sample_indices] if N > max_samples else identity_encoding
    neighbor_identity = identity_encoding[neighbor_indices]

    identity_diff = F.mse_loss(
        sample_identity.unsqueeze(1).expand(-1, k, -1),
        neighbor_identity,
        reduction='none'
    ).mean(dim=2)  # (N_sample, k)

    if same_id_mask.sum() == 0:
        identity_smooth_loss = torch.tensor(0.0).cuda()
    else:
        identity_smooth_loss = identity_diff[same_id_mask].mean()

    # Combined loss
    total_loss = (
        1.0 * normal_loss +
        0.5 * scale_loss +
        0.5 * identity_smooth_loss
    )

    return lambda_val * total_loss


def boundary_refinement_loss(
    rendered_objects: torch.Tensor,
    gt_objects: torch.Tensor,
    edge_weight: float = 2.0,
    lambda_val: float = 1.0
) -> torch.Tensor:
    
    C, H, W = rendered_objects.shape

    # Convert GT to one-hot
    gt_one_hot = F.one_hot(gt_objects.long(), num_classes=C).permute(2, 0, 1).float()

    # Detect edges in GT using Sobel
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).cuda()
    sobel_y = sobel_x.T

    # Apply Sobel to each channel (use max across channels)
    edge_map = torch.zeros(H, W).cuda()

    for c in range(C):
        gt_channel = gt_one_hot[c:c+1].unsqueeze(0)  # (1, 1, H, W)

        edge_x = F.conv2d(gt_channel, sobel_x.view(1, 1, 3, 3), padding=1)
        edge_y = F.conv2d(gt_channel, sobel_y.view(1, 1, 3, 3), padding=1)

        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze()
        edge_map = torch.max(edge_map, edge_magnitude)

    # Normalize edge map to [0, 1]
    edge_map = edge_map / (edge_map.max() + 1e-6)

    # Weight map: higher weight at boundaries
    weight_map = 1.0 + (edge_weight - 1.0) * edge_map  # (H, W)

    # Weighted cross-entropy loss
    # Compute per-pixel loss
    log_probs = F.log_softmax(rendered_objects, dim=0)  # (C, H, W)
    pixel_loss = -torch.sum(gt_one_hot * log_probs, dim=0)  # (H, W)

    # Apply weight map
    weighted_loss = (pixel_loss * weight_map).mean()

    return lambda_val * weighted_loss


def temporal_consistency_loss(
    features_t: torch.Tensor,
    features_t_minus_1: torch.Tensor,
    optical_flow: Optional[torch.Tensor] = None,
    lambda_val: float = 1.0
) -> torch.Tensor:
    
    if optical_flow is not None:
        # Warp features_t_minus_1 to time t using optical flow
        C, H, W = features_t.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).cuda()  # (2, H, W)

        # Add optical flow
        flow_normalized = optical_flow / torch.tensor([W, H]).view(2, 1, 1).cuda()
        warped_grid = grid + flow_normalized

        # Warp features
        warped_grid = warped_grid.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)
        features_t_minus_1_warped = F.grid_sample(
            features_t_minus_1.unsqueeze(0),
            warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(0)

        # Temporal consistency loss
        loss = F.mse_loss(features_t, features_t_minus_1_warped)

    else:
        # Simple temporal consistency (no flow)
        loss = F.mse_loss(features_t, features_t_minus_1)

    return lambda_val * loss


def plane_normal_consistency_loss(
    gaussians_xyz: torch.Tensor,
    predicted_ids: torch.Tensor,
    k: int = 12,
    max_samples: int = 2000,
    lambda_val: float = 1.0
) -> torch.Tensor:
    
    if gaussians_xyz.numel() == 0:
        return torch.tensor(0.0, device=gaussians_xyz.device)

    device = gaussians_xyz.device
    N = gaussians_xyz.shape[0]
    max_samples = min(max_samples, N)

    sample_idx = torch.randperm(N, device=device)[:max_samples]
    sample_pts = gaussians_xyz[sample_idx]
    sample_ids = predicted_ids[sample_idx]

    chunk_size = 500  # Process 500 samples at a time
    nn_idx_list = []

    for i in range(0, max_samples, chunk_size):
        end_idx = min(i + chunk_size, max_samples)
        chunk_pts = sample_pts[i:end_idx]

        dists_chunk = torch.cdist(chunk_pts, gaussians_xyz)
        _, nn_idx_chunk = torch.topk(dists_chunk, k + 1, largest=False, dim=1)
        nn_idx_list.append(nn_idx_chunk[:, 1:])

        del dists_chunk  # Free memory immediately

    nn_idx = torch.cat(nn_idx_list, dim=0)

    neighbors = gaussians_xyz[nn_idx]
    centered = neighbors - sample_pts.unsqueeze(1)

    cov = torch.matmul(centered.transpose(1, 2), centered) / k
    eigvals, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    offsets = torch.abs(torch.sum(centered * normals.unsqueeze(1), dim=2))
    neighbor_ids = predicted_ids[nn_idx]
    same_mask = neighbor_ids == sample_ids.unsqueeze(1)

    if same_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = offsets[same_mask].mean()
    return lambda_val * loss


def laplacian_smoothness_loss(
    gaussians_xyz: torch.Tensor,
    predicted_ids: torch.Tensor,
    k: int = 12,
    max_samples: int = 2000,
    lambda_val: float = 1.0
) -> torch.Tensor:
    if gaussians_xyz.numel() == 0:
        return torch.tensor(0.0, device=gaussians_xyz.device)

    device = gaussians_xyz.device
    N = gaussians_xyz.shape[0]
    max_samples = min(max_samples, N)

    sample_idx = torch.randperm(N, device=device)[:max_samples]
    sample_pts = gaussians_xyz[sample_idx]
    sample_ids = predicted_ids[sample_idx]

    chunk_size = 500  # Process 500 samples at a time
    nn_idx_list = []

    for i in range(0, max_samples, chunk_size):
        end_idx = min(i + chunk_size, max_samples)
        chunk_pts = sample_pts[i:end_idx]

        dists_chunk = torch.cdist(chunk_pts, gaussians_xyz)
        _, nn_idx_chunk = torch.topk(dists_chunk, k + 1, largest=False, dim=1)
        nn_idx_list.append(nn_idx_chunk[:, 1:])

        del dists_chunk  # Free memory immediately

    nn_idx = torch.cat(nn_idx_list, dim=0)

    neighbors = gaussians_xyz[nn_idx]
    neighbor_ids = predicted_ids[nn_idx]
    same_mask = neighbor_ids == sample_ids.unsqueeze(1)

    same_counts = same_mask.sum(dim=1, keepdim=True).float()
    valid_mask = same_counts.squeeze(1) > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    same_counts = torch.clamp(same_counts, min=1.0)
    barycenter = torch.sum(neighbors * same_mask.unsqueeze(2), dim=1) / same_counts
    deltas = torch.norm(sample_pts - barycenter, dim=1)
    loss = deltas[valid_mask].mean()
    return lambda_val * loss


def opacity_binarization_loss(
    opacity: torch.Tensor,
    lambda_val: float = 1.0
) -> torch.Tensor:
    if opacity.numel() == 0:
        return torch.tensor(0.0, device=opacity.device)

    opacity = torch.clamp(opacity, 1e-4, 1.0 - 1e-4)
    binary_term = (opacity * (1.0 - opacity)).mean()
    return lambda_val * binary_term


def enhanced_gaussian_grouping_loss(
    # Rendering outputs
    rendered_image: torch.Tensor,
    rendered_objects: torch.Tensor,
    # Ground truth
    gt_image: torch.Tensor,
    gt_objects: torch.Tensor,
    # Gaussian properties
    gaussians_xyz: torch.Tensor,
    gaussians_scale: torch.Tensor,
    identity_encoding: torch.Tensor,
    dino_features_fine: torch.Tensor,
    dino_features_coarse: Optional[torch.Tensor] = None,
    # Predictions
    predicted_ids: Optional[torch.Tensor] = None,
    # Loss weights
    weights: Optional[dict] = None,
    # Optional
    prev_features: Optional[torch.Tensor] = None,
    optical_flow: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    
    # Default weights
    if weights is None:
        weights = {
            'rgb': 1.0,
            'object': 1.0,
            'semantic': 0.1,
            'geometric': 0.05,
            'boundary': 0.5,
            'temporal': 0.1
        }

    loss_dict = {}

    # 1. RGB reconstruction loss
    from utils.loss_utils import l1_loss, ssim

    loss_rgb = l1_loss(rendered_image, gt_image)
    loss_ssim = 1.0 - ssim(rendered_image, gt_image)
    loss_dict['rgb'] = weights['rgb'] * (0.8 * loss_rgb + 0.2 * loss_ssim)

    # 2. Object classification loss
    loss_obj = F.cross_entropy(
        rendered_objects.unsqueeze(0),
        gt_objects.unsqueeze(0).long()
    )
    loss_dict['object'] = weights['object'] * loss_obj

    # 3. Multi-scale semantic cohesion loss
    if dino_features_fine is not None:
        loss_semantic = multiscale_semantic_cohesion_loss(
            gaussians_xyz,
            identity_encoding,
            dino_features_fine,
            dino_features_coarse,
            k=10,
            spatial_threshold=0.5
        )
        loss_dict['semantic'] = weights['semantic'] * loss_semantic
    else:
        loss_dict['semantic'] = torch.tensor(0.0).cuda()

    # 4. Geometric constraint loss
    if predicted_ids is not None:
        loss_geometric = geometric_constraint_loss(
            gaussians_xyz,
            gaussians_scale,
            identity_encoding,
            predicted_ids,
            k=10
        )
        loss_dict['geometric'] = weights['geometric'] * loss_geometric
    else:
        loss_dict['geometric'] = torch.tensor(0.0).cuda()

    # 5. Boundary refinement loss
    loss_boundary = boundary_refinement_loss(
        rendered_objects,
        gt_objects,
        edge_weight=2.0
    )
    loss_dict['boundary'] = weights['boundary'] * loss_boundary

    # 6. Temporal consistency loss (if applicable)
    if prev_features is not None:
        loss_temporal = temporal_consistency_loss(
            identity_encoding,
            prev_features,
            optical_flow
        )
        loss_dict['temporal'] = weights['temporal'] * loss_temporal
    else:
        loss_dict['temporal'] = torch.tensor(0.0).cuda()

    # Total loss
    total_loss = sum(loss_dict.values())

    return total_loss, loss_dict

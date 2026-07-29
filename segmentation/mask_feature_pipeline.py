from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 may be absent in minimal setups
    cv2 = None

from utils.dino_utils import DINOv2FeatureExtractor

from .edge_utils import LoGEdgeDetector
from .sam_refiner import SAMMaskGenerator
from .dino_cache import load_feature_from_cache, save_feature_to_cache


@dataclass
class MaskFeatureConfig:
    sam_checkpoint: str
    
    # --- DEFAULT ARGUMENTS FOLLOW ---
    sam_model_type: str = "vit_h"
    device: str = "cuda"
    sam_device: Optional[str] = None
    sam_fallback_device: str = "cpu"
    dino_model_name: str = "dinov2_vits14_reg_lc"
    dino_fallback_models: Optional[List[str]] = None
    min_mask_area: int = 512

    # === LoG EDGE DETECTION PARAMETERS ===
    log_sigma: float = 2.0
    laplacian_ksize: int = 3
    
    # === MERGING HYPERPARAMETERS (DEPTH-AWARE CONSERVATIVE) ===
    feature_weight: float = 1.0
    feature_sim_threshold: float = 0.75         # Higher threshold = less merging, more separation
    edge_strength_threshold: float = 0.25       # Lower = respect more edges
    edge_penalty: float = 0.8                   # Higher penalty = prevent merging across edges

    adjacency_dilation: int = 2                 # Smaller dilation = stricter adjacency
    min_contact_ratio: float = 0.01             # Higher = require more contact to merge
    max_merge_iterations: int = 500             # Fewer iterations
    adjacency_max_bbox_gap: int = 4             # Smaller gap = stricter adjacency
    feature_margin: float = 0.10                # Less tolerant = more strict feature matching

    # === LEARNABLE-READY MERGE SCORER PARAMETERS ===
    use_merge_scorer: bool = False
    merge_score_threshold: float = 0.0
    merge_scorer_bias: float = 0.0
    merge_scorer_weights: Optional[Dict[str, float]] = None
    record_merge_candidates: bool = False
    max_merge_candidate_records: int = 20000

    # === OPEN-VOCABULARY LANGUAGE MERGE EVIDENCE ===
    language_queries: Optional[List[str]] = None
    language_model_name: str = "ViT-B-32"
    language_pretrained: str = "laion2b_s34b_b79k"
    language_device: str = "cpu"
    language_score_prior: float = 0.5
    language_weight: float = 0.5
    language_crop_pad: float = 0.08
    language_background: str = "blur"
    language_batch_size: int = 32
    use_language_judge: bool = False
    language_judge_prior: float = 0.5
    language_judge_weight: float = 0.5
    language_judge_min_group_masks: int = 2

    # === SAM COVERAGE RECOVERY PARAMETERS ===
    recover_unassigned_regions: bool = False
    recovery_min_area: int = 256
    recovery_max_area_ratio: float = 0.15
    recovery_max_regions: int = 64
    recovery_include_border_components: bool = False
    use_recovery_scorer: bool = False
    recovery_score_threshold: float = 0.0
    recovery_scorer_bias: float = 0.0
    recovery_scorer_weights: Optional[Dict[str, float]] = None
    recovery_feature_sim_prior: float = 0.55
    recovery_edge_prior: float = 0.25
    recovery_contact_prior: float = 0.02
    recovery_depth_diff_prior: float = 0.15
    recovery_bbox_fill_prior: float = 0.02

    # SAM kwargs for finer initial segmentation
    sam_generator_kwargs: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "points_per_side": 32,                  # More points = finer segmentation
        "pred_iou_thresh": 0.88,                # Higher = better quality masks
        "stability_score_thresh": 0.95,         # Higher = more stable masks
        "min_mask_region_area": 400,            # Lower = keep smaller regions
    })
    
    dino_max_long_edge: int = 1600
    dino_tile_size: int = 0
    dino_tile_stride: Optional[int] = None
    dino_cache_dir: Optional[str] = None
    dino_cache_precision: str = "float32"
    sam_max_long_edge: int = 2048
    
    # === DEPTH-AWARE MERGING PARAMETERS (STRICT) ===
    use_depth: bool = True
    depth_method: str = "midas"
    depth_cache_dir: Optional[str] = None
    depth_diff_threshold: float = 0.15          # LOW tolerance = prevent merging objects at different depths
    depth_boundary_threshold: float = 0.25      # Prevent merging across depth boundaries
    depth_weight: float = 0.5                   # HIGH depth influence = respect depth differences
    depth_boundary_weight: float = 1.0          # Strong penalty for crossing depth boundaries
    depth_gradient_sigma: float = 1.0           # Smaller sigma = sharper depth boundaries

@dataclass
class MaskRegion:
    mask: np.ndarray
    feature: torch.Tensor
    area: int
    bbox: Tuple[int, int, int, int]
    region_id: int
    source_ids: List[int] = field(default_factory=list)
    mean_depth: float = 0.0
    language_score: float = 0.0
    language_query_scores: Dict[str, float] = field(default_factory=dict)
    language_group_scores: Dict[str, float] = field(default_factory=dict)


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return x_min, y_min, x_max, y_max


def _binary_dilate_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    h, w = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy : dy + h, dx : dx + w]
    return result


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        result = _binary_dilate_once(result)
    return result


def _mask_union(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    covered = np.zeros(shape, dtype=bool)
    for mask in masks:
        if mask.shape != shape:
            raise ValueError(f"Mask shape {mask.shape} does not match target shape {shape}.")
        covered |= mask.astype(bool)
    return covered


def _component_records(mask: np.ndarray) -> List[Tuple[int, int, int, int, int, np.ndarray]]:
    """Return connected components as (area, x0, y0, x1, y1, component_mask)."""
    if not mask.any():
        return []

    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        records = []
        for label in range(1, num_labels):
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            records.append((area, x0, y0, x0 + width - 1, y0 + height - 1, labels == label))
        return records

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    records = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            coords = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            ys = np.array([coord[0] for coord in coords], dtype=np.int32)
            xs = np.array([coord[1] for coord in coords], dtype=np.int32)
            component = np.zeros_like(mask, dtype=bool)
            component[ys, xs] = True
            records.append((
                int(len(coords)),
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
                component,
            ))
    return records


def _mask_recovery_report(enabled: bool, masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> Dict[str, object]:
    image_area = max(1, int(shape[0]) * int(shape[1]))
    valid_masks = [mask.astype(bool) for mask in masks if mask.shape == shape]
    covered = _mask_union(valid_masks, shape) if valid_masks else np.zeros(shape, dtype=bool)
    return {
        "enabled": bool(enabled),
        "input_masks": int(len(valid_masks)),
        "recovered_masks": 0,
        "candidate_components": 0,
        "coverage_before": float(covered.sum() / image_area),
        "coverage_after": float(covered.sum() / image_area),
        "unassigned_pixels_before": int((~covered).sum()),
        "unassigned_pixels_after": int((~covered).sum()),
        "selected_components": [],
        "rejected_components": {},
    }


def recover_unassigned_region_masks(
    masks: Sequence[np.ndarray],
    shape: Tuple[int, int],
    *,
    min_area: int = 256,
    max_area_ratio: float = 0.15,
    max_regions: int = 64,
    include_border_components: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """Append mask candidates for SAM-uncovered image regions.

    MCM cannot merge or split pixels that never enter the initial mask graph.
    This recovery stage adds connected uncovered components as low-assumption
    mask candidates, so later DINO/depth/edge evidence can decide whether they
    should merge into neighboring objects.
    """
    height, width = shape
    image_area = max(1, height * width)
    valid_masks = [mask.astype(bool) for mask in masks if mask.shape == shape]
    covered_before = _mask_union(valid_masks, shape) if valid_masks else np.zeros(shape, dtype=bool)
    uncovered = ~covered_before
    max_area = image_area if max_area_ratio <= 0 else int(round(image_area * max_area_ratio))
    max_area = max(int(min_area), max_area)

    candidates = []
    rejected = {
        "too_small": 0,
        "too_large": 0,
        "touches_border": 0,
        "isolated": 0,
    }
    has_covered_pixels = bool(covered_before.any())
    for area, x0, y0, x1, y1, component in _component_records(uncovered):
        touches_border = x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1
        if area < min_area:
            rejected["too_small"] += 1
            continue
        if area > max_area:
            rejected["too_large"] += 1
            continue
        if touches_border and not include_border_components:
            rejected["touches_border"] += 1
            continue
        if has_covered_pixels:
            neighbor_contact = np.logical_and(binary_dilation(component, iterations=1), covered_before).any()
            if not neighbor_contact:
                rejected["isolated"] += 1
                continue
        candidates.append({
            "area": int(area),
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "touches_border": bool(touches_border),
            "mask": component,
        })

    candidates.sort(key=lambda item: item["area"], reverse=True)
    selected = candidates[: max(0, int(max_regions))]
    recovered_masks = [item["mask"] for item in selected]
    output_masks = valid_masks + recovered_masks
    covered_after = _mask_union(output_masks, shape) if output_masks else covered_before

    report = {
        "enabled": True,
        "input_masks": int(len(valid_masks)),
        "recovered_masks": int(len(recovered_masks)),
        "candidate_components": int(len(candidates)),
        "coverage_before": float(covered_before.sum() / image_area),
        "coverage_after": float(covered_after.sum() / image_area),
        "unassigned_pixels_before": int((~covered_before).sum()),
        "unassigned_pixels_after": int((~covered_after).sum()),
        "min_area": int(min_area),
        "max_area_ratio": float(max_area_ratio),
        "max_regions": int(max_regions),
        "include_border_components": bool(include_border_components),
        "selected_components": [
            {
                "area": int(item["area"]),
                "bbox": list(item["bbox"]),
                "touches_border": bool(item["touches_border"]),
            }
            for item in selected
        ],
        "rejected_components": rejected,
    }
    return output_masks, report


class MaskFeatureLifter:
    def __init__(
        self,
        extractor: DINOv2FeatureExtractor,
        max_long_edge: int,
        tile_size: int,
        tile_stride: Optional[int],
    ):
        self.extractor = extractor
        self.max_long_edge = max_long_edge
        self.tile_size = max(0, tile_size)
        self.tile_stride = tile_stride if tile_stride is not None else tile_size
        self.patch_size = getattr(extractor, "patch_size", 14)

    def compute_feature_map(self, image: np.ndarray) -> torch.Tensor:
        processed = self._maybe_downscale(image)
        padded, pad_h, pad_w = self._pad_to_patch_multiple(processed)

        with torch.no_grad():
            if self.tile_size > 0 and (
                padded.shape[0] > self.tile_size or padded.shape[1] > self.tile_size
            ):
                features = self._extract_features_tiled(padded)
            else:
                features = self.extractor.extract_features(padded)

        features = self._remove_feature_padding(features, pad_h, pad_w)
        result = features.detach().cpu()

        # Clear GPU memory
        del features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def build_regions(
        self,
        masks: Sequence[np.ndarray],
        feature_map: torch.Tensor,
        depth_map: Optional[np.ndarray] = None,
        language_scores: Optional[Sequence[float]] = None,
        language_query_scores: Optional[Sequence[Dict[str, float]]] = None,
        language_group_scores: Optional[Sequence[Dict[str, float]]] = None,
    ) -> List[MaskRegion]:
        if not masks:
            return []

        target_shape = masks[0].shape
        upsampled = self._upsample_feature_map(feature_map, target_shape)
        regions: List[MaskRegion] = []

        for idx, mask in enumerate(masks):
            area = int(mask.sum())
            if area == 0:
                continue
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(-1)
            pooled = (upsampled * mask_tensor).sum(dim=(0, 1)) / (area + 1e-6)

            # Compute mean depth for this region
            mean_depth = 0.0
            if depth_map is not None:
                mean_depth = float(depth_map[mask].mean())
            language_score = 0.0
            if language_scores is not None and idx < len(language_scores):
                language_score = float(language_scores[idx])
            query_scores: Dict[str, float] = {}
            if language_query_scores is not None and idx < len(language_query_scores):
                query_scores = {str(key): float(value) for key, value in language_query_scores[idx].items()}
            group_scores: Dict[str, float] = {}
            if language_group_scores is not None and idx < len(language_group_scores):
                group_scores = {str(key): float(value) for key, value in language_group_scores[idx].items()}

            regions.append(
                MaskRegion(
                    mask=mask,
                    feature=pooled,
                    area=area,
                    bbox=_compute_bbox(mask),
                    region_id=idx,
                    source_ids=[idx],
                    mean_depth=mean_depth,
                    language_score=language_score,
                    language_query_scores=query_scores,
                    language_group_scores=group_scores,
                )
            )
        return regions

    def _maybe_downscale(self, image: np.ndarray) -> np.ndarray:
        if self.max_long_edge <= 0:
            return image
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.max_long_edge:
            return image
        scale = self.max_long_edge / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resized = Image.fromarray(image).resize(new_size, Image.BICUBIC)
        return np.array(resized)

    def _pad_to_patch_multiple(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        height, width = image.shape[:2]
        pad_h = (math.ceil(height / self.patch_size) * self.patch_size) - height
        pad_w = (math.ceil(width / self.patch_size) * self.patch_size) - width
        if pad_h == 0 and pad_w == 0:
            return image, 0, 0
        padded = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )
        return padded, pad_h, pad_w

    def _remove_feature_padding(self, feature_map: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return feature_map
        pad_tokens_h = pad_h // self.patch_size
        pad_tokens_w = pad_w // self.patch_size
        h, w, _ = feature_map.shape
        end_h = h - pad_tokens_h if pad_tokens_h > 0 else h
        end_w = w - pad_tokens_w if pad_tokens_w > 0 else w
        return feature_map[:end_h, :end_w]

    def _extract_features_tiled(self, image: np.ndarray) -> torch.Tensor:
        tile_size = max(self.patch_size, (self.tile_size // self.patch_size) * self.patch_size)
        stride = self.tile_stride or tile_size
        stride = max(self.patch_size, (stride // self.patch_size) * self.patch_size)

        height, width = image.shape[:2]
        tile_size = min(tile_size, height, width)

        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        feature_dim = self.extractor.feature_dim

        feature_sum = torch.zeros(h_tokens, w_tokens, feature_dim, dtype=torch.float32)
        counts = torch.zeros(h_tokens, w_tokens, dtype=torch.float32)

        y_positions = self._generate_positions(height, tile_size, stride)
        x_positions = self._generate_positions(width, tile_size, stride)

        for y0 in y_positions:
            for x0 in x_positions:
                y1 = y0 + tile_size
                x1 = x0 + tile_size
                tile = image[y0:y1, x0:x1]
                tile_features = self.extractor.extract_features(tile).detach().cpu()
                th, tw, _ = tile_features.shape

                y_patch0 = y0 // self.patch_size
                x_patch0 = x0 // self.patch_size

                feature_sum[y_patch0 : y_patch0 + th, x_patch0 : x_patch0 + tw] += tile_features
                counts[y_patch0 : y_patch0 + th, x_patch0 : x_patch0 + tw] += 1.0

                # Clear memory after each tile
                del tile_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        counts = torch.clamp(counts, min=1.0).unsqueeze(-1)
        return feature_sum / counts

    @staticmethod
    def _generate_positions(length: int, tile: int, stride: int) -> List[int]:
        positions = list(range(0, max(1, length - tile + 1), stride))
        if positions[-1] != length - tile:
            positions.append(max(0, length - tile))
        return sorted(set(positions))

    @staticmethod
    def _upsample_feature_map(feature_map: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        tensor = feature_map.permute(2, 0, 1).unsqueeze(0)  # (1, D, Hf, Wf)
        upsampled = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
        return upsampled.squeeze(0).permute(1, 2, 0)  # (H, W, D)


@dataclass
class RecoveryEvidence:
    feature_similarity: float
    edge_strength: float
    contact_ratio: float
    area_ratio: float
    bbox_fill_ratio: float
    depth_difference: float = 0.0
    depth_available: bool = False
    has_neighbor: bool = False
    touches_border: bool = False

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "feature_similarity": float(self.feature_similarity),
            "edge_strength": float(self.edge_strength),
            "contact_ratio": float(self.contact_ratio),
            "area_ratio": float(self.area_ratio),
            "bbox_fill_ratio": float(self.bbox_fill_ratio),
            "depth_difference": float(self.depth_difference),
            "depth_available": bool(self.depth_available),
            "has_neighbor": bool(self.has_neighbor),
            "touches_border": bool(self.touches_border),
        }


class RecoveryCandidateScorer:
    """Linear scorer for SAM-uncovered mask candidates.

    The candidate generator is intentionally simple: it proposes connected
    uncovered components. This scorer decides whether each proposal is plausible
    using DINO similarity, LoG/depth boundary agreement, contact with existing
    SAM regions, and component size.
    """

    DEFAULT_WEIGHTS = {
        "feature": 1.0,
        "edge": 0.7,
        "contact": 0.3,
        "area": 0.05,
        "bbox_fill": 0.05,
        "depth": 0.3,
        "neighbor": 0.5,
        "border": 0.2,
    }

    def __init__(
        self,
        *,
        feature_prior: float = 0.55,
        edge_prior: float = 0.25,
        contact_prior: float = 0.02,
        area_prior: float = 0.15,
        bbox_fill_prior: float = 0.02,
        depth_diff_prior: Optional[float] = 0.15,
        score_threshold: float = 0.0,
        bias: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.feature_prior = float(feature_prior)
        self.edge_prior = float(edge_prior)
        self.contact_prior = float(contact_prior)
        self.area_prior = float(area_prior)
        self.bbox_fill_prior = float(bbox_fill_prior)
        self.depth_diff_prior = None if depth_diff_prior is None else float(depth_diff_prior)
        self.score_threshold = float(score_threshold)
        self.bias = float(bias)
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            for key, value in weights.items():
                if key not in self.weights:
                    raise ValueError(f"Unknown recovery scorer weight: {key}")
                self.weights[key] = float(value)

    def score(self, evidence: RecoveryEvidence) -> float:
        score = self.bias
        score += self.weights["feature"] * (float(evidence.feature_similarity) - self.feature_prior)
        score += self.weights["edge"] * (self.edge_prior - float(evidence.edge_strength))
        score += self.weights["contact"] * (float(evidence.contact_ratio) - self.contact_prior)
        score += self.weights["area"] * (self.area_prior - float(evidence.area_ratio))
        score += self.weights["bbox_fill"] * (float(evidence.bbox_fill_ratio) - self.bbox_fill_prior)
        if evidence.depth_available and self.depth_diff_prior is not None:
            denom = max(self.depth_diff_prior, 1e-6)
            score += self.weights["depth"] * ((self.depth_diff_prior - float(evidence.depth_difference)) / denom)
        if evidence.has_neighbor:
            score += self.weights["neighbor"]
        else:
            score -= self.weights["neighbor"]
        if evidence.touches_border:
            score -= self.weights["border"]
        return float(score)

    def should_accept(self, evidence: RecoveryEvidence) -> tuple[bool, float]:
        score = self.score(evidence)
        return score > self.score_threshold, score

    def describe(self) -> Dict[str, object]:
        return {
            "feature_prior": self.feature_prior,
            "edge_prior": self.edge_prior,
            "contact_prior": self.contact_prior,
            "area_prior": self.area_prior,
            "bbox_fill_prior": self.bbox_fill_prior,
            "depth_diff_prior": self.depth_diff_prior,
            "score_threshold": self.score_threshold,
            "bias": self.bias,
            "weights": dict(self.weights),
        }


@dataclass
class MergeEvidence:
    feature_similarity: float
    edge_strength: float
    contact_ratio: float
    bbox_gap: float
    area_balance: float
    depth_difference: float = 0.0
    depth_boundary_strength: float = 0.0
    depth_available: bool = False
    depth_boundary_available: bool = False
    language_affinity: float = 0.0
    left_language_score: float = 0.0
    right_language_score: float = 0.0
    language_available: bool = False
    language_group_affinity: float = 0.0
    left_language_group_score: float = 0.0
    right_language_group_score: float = 0.0
    language_group_query: str = ""
    language_group_available: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_similarity": float(self.feature_similarity),
            "edge_strength": float(self.edge_strength),
            "contact_ratio": float(self.contact_ratio),
            "bbox_gap": float(self.bbox_gap),
            "area_balance": float(self.area_balance),
            "depth_difference": float(self.depth_difference),
            "depth_boundary_strength": float(self.depth_boundary_strength),
            "depth_available": bool(self.depth_available),
            "depth_boundary_available": bool(self.depth_boundary_available),
            "language_affinity": float(self.language_affinity),
            "left_language_score": float(self.left_language_score),
            "right_language_score": float(self.right_language_score),
            "language_available": bool(self.language_available),
            "language_group_affinity": float(self.language_group_affinity),
            "left_language_group_score": float(self.left_language_group_score),
            "right_language_group_score": float(self.right_language_group_score),
            "language_group_query": str(self.language_group_query),
            "language_group_available": bool(self.language_group_available),
        }


class MergeEvidenceScorer:
    """Linear evidence scorer for mask-pair merging.

    This replaces independent hard thresholds with a single calibrated merge
    score. The weights are plain floats now, but the interface mirrors a
    learnable linear layer so we can later fit them from pseudo labels or
    view-consistency losses without changing the merger logic.
    """

    DEFAULT_WEIGHTS = {
        "feature": 1.0,
        "edge": 1.0,
        "contact": 0.2,
        "bbox_gap": 0.05,
        "area_balance": 0.05,
        "depth": 0.5,
        "depth_boundary": 1.0,
        "language": 0.5,
        "language_group": 0.5,
    }

    def __init__(
        self,
        *,
        feature_prior: float,
        edge_prior: float,
        contact_prior: float,
        depth_diff_prior: Optional[float],
        depth_boundary_prior: Optional[float],
        max_bbox_gap: int,
        language_prior: Optional[float] = None,
        language_group_prior: Optional[float] = None,
        score_threshold: float = 0.0,
        bias: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.feature_prior = float(feature_prior)
        self.edge_prior = float(edge_prior)
        self.contact_prior = float(contact_prior)
        self.depth_diff_prior = None if depth_diff_prior is None else float(depth_diff_prior)
        self.depth_boundary_prior = None if depth_boundary_prior is None else float(depth_boundary_prior)
        self.max_bbox_gap = max(1, int(max_bbox_gap))
        self.language_prior = None if language_prior is None else float(language_prior)
        self.language_group_prior = None if language_group_prior is None else float(language_group_prior)
        self.score_threshold = float(score_threshold)
        self.bias = float(bias)
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            for key, value in weights.items():
                if key not in self.weights:
                    raise ValueError(f"Unknown merge scorer weight: {key}")
                self.weights[key] = float(value)

    def score(self, evidence: MergeEvidence) -> float:
        bbox_compat = 1.0 - min(max(float(evidence.bbox_gap), 0.0) / float(self.max_bbox_gap), 1.0)
        area_balance = max(0.0, min(float(evidence.area_balance), 1.0))
        contact = max(0.0, min(float(evidence.contact_ratio), 1.0))
        score = self.bias
        score += self.weights["feature"] * (float(evidence.feature_similarity) - self.feature_prior)
        score += self.weights["edge"] * (self.edge_prior - float(evidence.edge_strength))
        score += self.weights["contact"] * (contact - self.contact_prior)
        score += self.weights["bbox_gap"] * bbox_compat
        score += self.weights["area_balance"] * (area_balance - 0.5)
        if evidence.depth_available and self.depth_diff_prior is not None:
            denom = max(self.depth_diff_prior, 1e-6)
            score += self.weights["depth"] * ((self.depth_diff_prior - float(evidence.depth_difference)) / denom)
        if evidence.depth_boundary_available and self.depth_boundary_prior is not None:
            score += self.weights["depth_boundary"] * (
                self.depth_boundary_prior - float(evidence.depth_boundary_strength)
            )
        if evidence.language_available and self.language_prior is not None:
            score += self.weights["language"] * (float(evidence.language_affinity) - self.language_prior)
        if evidence.language_group_available and self.language_group_prior is not None:
            score += self.weights["language_group"] * (
                float(evidence.language_group_affinity) - self.language_group_prior
            )
        return float(score)

    def should_merge(self, evidence: MergeEvidence) -> tuple[bool, float]:
        score = self.score(evidence)
        return score > self.score_threshold, score

    def describe(self) -> Dict[str, object]:
        return {
            "feature_prior": self.feature_prior,
            "edge_prior": self.edge_prior,
            "contact_prior": self.contact_prior,
            "depth_diff_prior": self.depth_diff_prior,
            "depth_boundary_prior": self.depth_boundary_prior,
            "max_bbox_gap": self.max_bbox_gap,
            "language_prior": self.language_prior,
            "language_group_prior": self.language_group_prior,
            "score_threshold": self.score_threshold,
            "bias": self.bias,
            "weights": dict(self.weights),
        }


class HierarchicalMaskMerger:
    def __init__(
        self,
        feature_sim_threshold: float,
        edge_strength_threshold: float,
        edge_penalty: float,
        adjacency_dilation: int,
        min_contact_ratio: float,
        max_merge_iterations: int,
        depth_diff_threshold: Optional[float] = None,
        depth_boundary_threshold: Optional[float] = None,
        depth_weight: float = 0.0,
        max_bbox_gap: int = 4,
        feature_margin: float = 0.1,
        depth_boundary_weight: float = 0.0,
        feature_weight: float = 1.0,
        use_scorer: bool = False,
        merge_score_threshold: float = 0.0,
        merge_scorer_bias: float = 0.0,
        merge_scorer_weights: Optional[Dict[str, float]] = None,
        record_candidate_evidence: bool = False,
        max_candidate_records: int = 20000,
        language_score_prior: Optional[float] = None,
        language_weight: float = 0.5,
        language_group_prior: Optional[float] = None,
        language_group_weight: float = 0.5,
    ) -> None:
        self.feature_sim_threshold = feature_sim_threshold
        self.edge_strength_threshold = edge_strength_threshold
        self.edge_penalty = edge_penalty
        self.edge_weight = edge_penalty
        self.feature_weight = feature_weight
        self.adjacency_dilation = adjacency_dilation
        self.min_contact_ratio = min_contact_ratio
        self.max_merge_iterations = max_merge_iterations
        self.max_bbox_gap = max(0, max_bbox_gap)
        self.feature_margin = feature_margin
        self.depth_diff_threshold = depth_diff_threshold
        self.depth_boundary_threshold = depth_boundary_threshold
        self.depth_weight = depth_weight
        self.depth_boundary_weight = depth_boundary_weight
        self.use_depth = depth_diff_threshold is not None
        self.use_scorer = bool(use_scorer)
        self.record_candidate_evidence = bool(record_candidate_evidence)
        self.max_candidate_records = max(0, int(max_candidate_records))
        self.language_score_prior = None if language_score_prior is None else float(language_score_prior)
        self.language_group_prior = None if language_group_prior is None else float(language_group_prior)
        self._candidate_records: List[Dict[str, object]] = []
        default_weights = {
            "feature": float(feature_weight),
            "edge": float(edge_penalty),
            "contact": 0.2,
            "bbox_gap": 0.05,
            "area_balance": 0.05,
            "depth": float(depth_weight),
            "depth_boundary": float(depth_boundary_weight),
            "language": float(language_weight),
            "language_group": float(language_group_weight),
        }
        if merge_scorer_weights:
            default_weights.update({key: float(value) for key, value in merge_scorer_weights.items()})
        self.scorer = MergeEvidenceScorer(
            feature_prior=feature_sim_threshold,
            edge_prior=edge_strength_threshold,
            contact_prior=min_contact_ratio,
            depth_diff_prior=depth_diff_threshold,
            depth_boundary_prior=depth_boundary_threshold,
            max_bbox_gap=max(1, max_bbox_gap),
            language_prior=language_score_prior,
            language_group_prior=language_group_prior,
            score_threshold=merge_score_threshold,
            bias=merge_scorer_bias,
            weights=default_weights,
        )
        self.last_merge_report: Dict[str, object] = {}

    def merge(
        self,
        regions: List[MaskRegion],
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        depth_gradients: Optional[np.ndarray] = None,
    ) -> List[MaskRegion]:
        merged_regions = list(regions)
        iteration = 0
        accepted = []
        self._candidate_records = []
        self.last_merge_report = {
            "initial_regions": len(regions),
            "use_scorer": self.use_scorer,
            "scorer": self.scorer.describe(),
            "accepted_merges": accepted,
        }

        while iteration < self.max_merge_iterations:
            candidate = self._select_best_pair(merged_regions, edge_map, depth_map, depth_gradients, iteration)
            if candidate is None:
                break

            i, j, score, evidence = candidate
            accepted.append({
                "iteration": int(iteration),
                "score": float(score),
                "left_sources": list(merged_regions[i].source_ids),
                "right_sources": list(merged_regions[j].source_ids),
                "evidence": evidence.to_dict(),
            })
            new_region = self._merge_pair(merged_regions[i], merged_regions[j], depth_map)
            merged_regions.pop(max(i, j))
            merged_regions.pop(min(i, j))
            merged_regions.append(new_region)
            iteration += 1

        self.last_merge_report["final_regions"] = len(merged_regions)
        self.last_merge_report["num_merges"] = len(accepted)
        if self.record_candidate_evidence:
            self.last_merge_report["candidate_evaluations"] = self._candidate_records
            self.last_merge_report["candidate_evaluations_truncated"] = (
                len(self._candidate_records) >= self.max_candidate_records
            )
        return merged_regions

    def _select_best_pair(
        self,
        regions: List[MaskRegion],
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        depth_gradients: Optional[np.ndarray] = None,
        iteration: int = 0,
    ) -> Optional[Tuple[int, int, float, MergeEvidence]]:
        best_pair: Optional[Tuple[int, int, float, MergeEvidence]] = None
        best_score = float("-inf")

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if not self._is_adjacent(regions[i], regions[j]):
                    continue

                evidence = self._build_evidence(regions[i], regions[j], edge_map, depth_map, depth_gradients)
                if evidence is None:
                    continue

                legacy_should_merge, legacy_score = self._legacy_should_merge(evidence)
                scorer_should_merge, scorer_score = self.scorer.should_merge(evidence)
                if self.record_candidate_evidence and len(self._candidate_records) < self.max_candidate_records:
                    self._candidate_records.append({
                        "iteration": int(iteration),
                        "left_sources": list(regions[i].source_ids),
                        "right_sources": list(regions[j].source_ids),
                        "legacy_label": bool(legacy_should_merge),
                        "legacy_score": float(legacy_score),
                        "scorer_label": bool(scorer_should_merge),
                        "scorer_score": float(scorer_score),
                        "evidence": evidence.to_dict(),
                    })

                if self.use_scorer:
                    should_merge, score = scorer_should_merge, scorer_score
                    if not should_merge:
                        continue
                else:
                    should_merge, score = legacy_should_merge, legacy_score
                    if not should_merge:
                        continue

                if score > best_score:
                    best_score = score
                    best_pair = (i, j, score, evidence)

        return best_pair

    def _build_evidence(
        self,
        a: MaskRegion,
        b: MaskRegion,
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray],
        depth_gradients: Optional[np.ndarray],
    ) -> Optional[MergeEvidence]:
        boundary_mask = self._shared_boundary_mask(a, b)
        if boundary_mask is None:
            return None
        depth_difference = abs(a.mean_depth - b.mean_depth) if depth_map is not None else 0.0
        depth_boundary_strength = 0.0
        depth_boundary_available = False
        if depth_gradients is not None:
            depth_boundary_strength = self._depth_boundary_strength(boundary_mask, depth_gradients)
            depth_boundary_available = True
        left_language = float(getattr(a, "language_score", 0.0))
        right_language = float(getattr(b, "language_score", 0.0))
        group_query, left_group, right_group, group_affinity = self._best_language_group_pair(a, b)
        return MergeEvidence(
            feature_similarity=self._feature_similarity(a, b),
            edge_strength=self._boundary_strength(boundary_mask, edge_map),
            contact_ratio=self._contact_ratio(a, b),
            bbox_gap=float(self._bbox_gap(a.bbox, b.bbox)),
            area_balance=min(a.area, b.area) / float(max(a.area, b.area, 1)),
            depth_difference=depth_difference,
            depth_boundary_strength=depth_boundary_strength,
            depth_available=depth_map is not None and self.use_depth,
            depth_boundary_available=depth_boundary_available and self.use_depth,
            language_affinity=min(left_language, right_language),
            left_language_score=left_language,
            right_language_score=right_language,
            language_available=self.language_score_prior is not None,
            language_group_affinity=group_affinity,
            left_language_group_score=left_group,
            right_language_group_score=right_group,
            language_group_query=group_query,
            language_group_available=self.language_group_prior is not None,
        )

    def _legacy_should_merge(self, evidence: MergeEvidence) -> tuple[bool, float]:
        if evidence.feature_similarity < self.feature_sim_threshold - self.feature_margin:
            return False, float("-inf")
        if evidence.edge_strength > self.edge_strength_threshold * (1.0 + self.feature_margin):
            return False, float("-inf")
        score = self.edge_weight * (self.edge_strength_threshold - evidence.edge_strength)
        score += self.feature_weight * (evidence.feature_similarity - self.feature_sim_threshold)
        if evidence.depth_available:
            if evidence.depth_difference >= float(self.depth_diff_threshold):
                return False, float("-inf")
            depth_score = self._compute_depth_score_from_difference(evidence.depth_difference)
            score += self.depth_weight * depth_score
            if evidence.depth_boundary_available and self.depth_boundary_weight > 0.0:
                if self.depth_boundary_threshold is not None:
                    excess = max(0.0, evidence.depth_boundary_strength - self.depth_boundary_threshold)
                else:
                    excess = evidence.depth_boundary_strength
                score -= self.depth_boundary_weight * excess
        return score > 0.0, float(score)

    def _is_adjacent(self, a: MaskRegion, b: MaskRegion) -> bool:
        if not self._bboxes_overlap(a.bbox, b.bbox):
            if self.max_bbox_gap == 0:
                return False
            gap = self._bbox_gap(a.bbox, b.bbox)
            return gap <= self.max_bbox_gap

        contact = self._contact_area(a.mask, b.mask)
        min_area = max(1, min(a.area, b.area))
        ratio = contact / float(min_area)
        if ratio >= self.min_contact_ratio:
            return True

        if self.max_bbox_gap > 0:
            gap = self._bbox_gap(a.bbox, b.bbox)
            return gap <= self.max_bbox_gap
        return False

    def _contact_area(self, mask_a: np.ndarray, mask_b: np.ndarray) -> int:
        dilated_a = binary_dilation(mask_a, iterations=self.adjacency_dilation)
        dilated_b = binary_dilation(mask_b, iterations=self.adjacency_dilation)
        return int(np.logical_and(dilated_a, dilated_b).sum())

    def _contact_ratio(self, a: MaskRegion, b: MaskRegion) -> float:
        min_area = max(1, min(a.area, b.area))
        return self._contact_area(a.mask, b.mask) / float(min_area)

    @staticmethod
    def _bboxes_overlap(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> bool:
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    @staticmethod
    def _feature_similarity(a: MaskRegion, b: MaskRegion) -> float:
        return torch.nn.functional.cosine_similarity(
            a.feature.unsqueeze(0), b.feature.unsqueeze(0), eps=1e-6
        ).item()

    def _shared_boundary_mask(self, a: MaskRegion, b: MaskRegion) -> Optional[np.ndarray]:
        dilated_a = binary_dilation(a.mask, iterations=1)
        dilated_b = binary_dilation(b.mask, iterations=1)
        boundary_contact = np.logical_and(dilated_a, dilated_b)
        if boundary_contact.sum() == 0:
            return None
        return boundary_contact

    def _boundary_strength(self, contact_mask: np.ndarray, edge_map: np.ndarray) -> float:
        if contact_mask.sum() == 0:
            return 0.0
        return float(edge_map[contact_mask].mean())

    def _depth_boundary_strength(self, contact_mask: np.ndarray, depth_gradients: np.ndarray) -> float:
        if depth_gradients is None or contact_mask.sum() == 0:
            return 0.0
        return float(depth_gradients[contact_mask].mean())

    def _merge_pair(self, a: MaskRegion, b: MaskRegion, depth_map: Optional[np.ndarray] = None) -> MaskRegion:
        merged_mask = np.logical_or(a.mask, b.mask)
        total_area = max(1, a.area + b.area)
        merged_feature = (a.feature * a.area + b.feature * b.area) / total_area
        merged_bbox = _compute_bbox(merged_mask)
        merged_ids = a.source_ids + b.source_ids
        merged_depth = 0.0
        if depth_map is not None:
            merged_depth = (a.mean_depth * a.area + b.mean_depth * b.area) / total_area
        merged_language = (a.language_score * a.area + b.language_score * b.area) / total_area
        merged_query_scores = self._merge_score_dicts(a.language_query_scores, b.language_query_scores, a.area, b.area)
        merged_group_scores = self._merge_score_dicts(a.language_group_scores, b.language_group_scores, a.area, b.area)
        return MaskRegion(
            mask=merged_mask,
            feature=merged_feature,
            area=int(merged_mask.sum()),
            bbox=merged_bbox,
            region_id=min(a.region_id, b.region_id),
            source_ids=merged_ids,
            mean_depth=merged_depth,
            language_score=float(merged_language),
            language_query_scores=merged_query_scores,
            language_group_scores=merged_group_scores,
        )

    @staticmethod
    def _merge_score_dicts(
        left: Dict[str, float],
        right: Dict[str, float],
        left_area: int,
        right_area: int,
    ) -> Dict[str, float]:
        total_area = max(1, int(left_area) + int(right_area))
        merged: Dict[str, float] = {}
        for key in set(left) | set(right):
            merged[key] = (
                float(left.get(key, 0.0)) * int(left_area)
                + float(right.get(key, 0.0)) * int(right_area)
            ) / float(total_area)
        return merged

    @staticmethod
    def _best_language_group_pair(a: MaskRegion, b: MaskRegion) -> Tuple[str, float, float, float]:
        left_scores = getattr(a, "language_group_scores", {}) or {}
        right_scores = getattr(b, "language_group_scores", {}) or {}
        best_query = ""
        best_left = 0.0
        best_right = 0.0
        best_affinity = 0.0
        for query in set(left_scores) & set(right_scores):
            left_value = float(left_scores.get(query, 0.0))
            right_value = float(right_scores.get(query, 0.0))
            affinity = min(left_value, right_value)
            if affinity > best_affinity:
                best_query = str(query)
                best_left = left_value
                best_right = right_value
                best_affinity = affinity
        return best_query, best_left, best_right, best_affinity

    def _compute_depth_score_from_difference(self, depth_diff: float) -> float:
        if self.depth_diff_threshold is None:
            return 0.0
        sigma = max(float(self.depth_diff_threshold) / 2.0, 1e-6)
        return float(np.exp(-float(depth_diff) / sigma))

    def _compute_depth_score(self, a: MaskRegion, b: MaskRegion, depth_map: np.ndarray) -> float:
        return self._compute_depth_score_from_difference(abs(a.mean_depth - b.mean_depth))

    @staticmethod
    def _bbox_gap(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> int:
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b
        x_gap = max(0, max(bx0 - ax1, ax0 - bx1))
        y_gap = max(0, max(by0 - ay1, ay0 - by1))
        return int(max(x_gap, y_gap))


class MaskFeaturePipeline:
    def __init__(
        self,
        config: MaskFeatureConfig,
        sam_generator: Optional[SAMMaskGenerator] = None,
        dino_extractor: Optional[DINOv2FeatureExtractor] = None,
    ) -> None:
        self.config = config
        sam_device = config.sam_device or config.device
        self.sam_generator = sam_generator or SAMMaskGenerator(
            checkpoint_path=config.sam_checkpoint,
            model_type=config.sam_model_type,
            device=sam_device,
            fallback_device=config.sam_fallback_device,
            generator_kwargs=config.sam_generator_kwargs,
            min_mask_area=config.min_mask_area,
        )
        self.edge_detector = LoGEdgeDetector(
            sigma=config.log_sigma,
            laplacian_ksize=config.laplacian_ksize
        )
        fallback_models = config.dino_fallback_models or ["dinov2_vits14_reg", "dinov2_vits14"]
        self.dino_extractor = dino_extractor or DINOv2FeatureExtractor(
            model_name=config.dino_model_name,
            device=config.device,
            fallback_models=fallback_models,
        )
        self.feature_lifter = MaskFeatureLifter(
            self.dino_extractor,
            max_long_edge=config.dino_max_long_edge,
            tile_size=config.dino_tile_size,
            tile_stride=config.dino_tile_stride,
        )
        self.feature_cache_dir = Path(config.dino_cache_dir).expanduser() if config.dino_cache_dir else None
        if self.feature_cache_dir:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sam_max_long_edge = config.sam_max_long_edge
        self.dino_cache_precision = config.dino_cache_precision

        # Initialize depth estimator or depth cache
        self.depth_estimator = None
        self.depth_cache_dir = None

        if config.use_depth:
            # Option 1: Use precomputed depth maps from cache
            if config.depth_cache_dir:
                self.depth_cache_dir = Path(config.depth_cache_dir).expanduser()
                if not self.depth_cache_dir.exists():
                    print(f"Warning: Depth cache directory not found: {config.depth_cache_dir}")
                    print("Falling back to on-the-fly depth estimation")
                    config.depth_cache_dir = None
                else:
                    print(f"✓ Using precomputed depth maps from: {config.depth_cache_dir}")

            # Option 2: Compute depth on-the-fly
            if not config.depth_cache_dir:
                try:
                    from utils.depth_estimator import DINOv2DepthEstimator
                    self.depth_estimator = DINOv2DepthEstimator(
                        method=config.depth_method,
                        device=config.device
                    )
                    print(f"✓ Depth estimation enabled ({config.depth_method})")
                except Exception as e:
                    print(f"Warning: Failed to initialize depth estimator: {e}")
                    print("Continuing without depth estimation")
                    config.use_depth = False

        self.merger = HierarchicalMaskMerger(
            feature_sim_threshold=config.feature_sim_threshold,
            edge_strength_threshold=config.edge_strength_threshold,
            edge_penalty=config.edge_penalty,
            adjacency_dilation=config.adjacency_dilation,
            min_contact_ratio=config.min_contact_ratio,
            max_merge_iterations=config.max_merge_iterations,
            depth_diff_threshold=config.depth_diff_threshold if config.use_depth else None,
            depth_boundary_threshold=config.depth_boundary_threshold if config.use_depth else None,
            depth_weight=config.depth_weight if config.use_depth else 0.0,
            max_bbox_gap=config.adjacency_max_bbox_gap,
            feature_margin=config.feature_margin,
            depth_boundary_weight=config.depth_boundary_weight if config.use_depth else 0.0,
            feature_weight=config.feature_weight,
            use_scorer=config.use_merge_scorer,
            merge_score_threshold=config.merge_score_threshold,
            merge_scorer_bias=config.merge_scorer_bias,
            merge_scorer_weights=config.merge_scorer_weights,
            record_candidate_evidence=config.record_merge_candidates,
            max_candidate_records=config.max_merge_candidate_records,
            language_score_prior=config.language_score_prior if config.language_queries else None,
            language_weight=config.language_weight,
            language_group_prior=(
                config.language_judge_prior
                if config.language_queries and config.use_language_judge
                else None
            ),
            language_group_weight=config.language_judge_weight,
        )
        self.recovery_scorer = RecoveryCandidateScorer(
            feature_prior=config.recovery_feature_sim_prior,
            edge_prior=config.recovery_edge_prior,
            contact_prior=config.recovery_contact_prior,
            area_prior=config.recovery_max_area_ratio,
            bbox_fill_prior=config.recovery_bbox_fill_prior,
            depth_diff_prior=config.recovery_depth_diff_prior if config.use_depth else None,
            score_threshold=config.recovery_score_threshold,
            bias=config.recovery_scorer_bias,
            weights=config.recovery_scorer_weights,
        )
        self.language_model = None
        self.language_preprocess = None
        self.language_tokenizer = None
        self.language_device = config.language_device
        if config.language_queries:
            self._initialize_language_model()

    def process_image(self, image: np.ndarray, image_name: Optional[str] = None) -> Dict[str, object]:
        """
        Runs SAM -> Canny edges -> feature lifting -> hierarchical merging on a single image.
        """
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sam_image, scale = self._prepare_sam_image(image)

        # Generate SAM masks with gradient disabled
        with torch.no_grad():
            sam_masks_small = self.sam_generator.generate(sam_image)

        # Clear cache after SAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        edge_map = self.edge_detector.compute(image)
        sam_masks = self._rescale_masks(sam_masks_small, image.shape[:2], scale)
        if self.config.recover_unassigned_regions:
            sam_masks, mask_recovery = recover_unassigned_region_masks(
                sam_masks,
                image.shape[:2],
                min_area=self.config.recovery_min_area,
                max_area_ratio=self.config.recovery_max_area_ratio,
                max_regions=self.config.recovery_max_regions,
                include_border_components=self.config.recovery_include_border_components,
            )
        else:
            mask_recovery = _mask_recovery_report(False, sam_masks, image.shape[:2])

        if not sam_masks:
            merge_report = {
                "initial_regions": 0,
                "final_regions": 0,
                "num_merges": 0,
                "mask_recovery": mask_recovery,
            }
            return {
                "image_name": image_name,
                "regions": [],
                "initial_masks": [],
                "edge_map": edge_map,
                "mask_recovery": mask_recovery,
                "merge_report": merge_report,
            }

        # Process DINO features with gradient disabled
        with torch.no_grad():
            feature_map = self._get_feature_map(image, image_name)

        # Clear cache after DINO
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract depth map if enabled
        depth_map = None
        if self.depth_cache_dir and image_name:
            # Load precomputed depth map
            depth_path = self.depth_cache_dir / f"{image_name}.npy"
            if depth_path.exists():
                depth_map = np.load(depth_path)
                print(f"  ✓ Loaded depth from cache: {depth_map.shape}, range [{depth_map.min():.3f}, {depth_map.max():.3f}]")
            else:
                print(f"  Warning: Depth map not found: {depth_path}")

        elif self.depth_estimator is not None:
            # Compute depth on-the-fly
            with torch.no_grad():
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                depth_map = self.depth_estimator.estimate(pil_image)
                print(f"  ✓ Depth extracted: {depth_map.shape}, range [{depth_map.min():.3f}, {depth_map.max():.3f}]")

            # Clear cache after depth estimation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        depth_gradients = None
        if depth_map is not None and self.config.use_depth:
            depth_gradients = self._compute_depth_gradients(depth_map)

        if self.config.use_recovery_scorer:
            sam_masks, mask_recovery = self._filter_recovered_masks_with_scorer(
                sam_masks,
                feature_map,
                edge_map,
                depth_map,
                mask_recovery,
                image.shape[:2],
            )

        language_scores, language_query_scores, language_group_scores, language_report = self._compute_language_scores(image, sam_masks)
        regions = self.feature_lifter.build_regions(
            sam_masks,
            feature_map,
            depth_map,
            language_scores=language_scores,
            language_query_scores=language_query_scores,
            language_group_scores=language_group_scores,
        )
        merged_regions = self.merger.merge(regions, edge_map, depth_map, depth_gradients)
        merge_report = dict(self.merger.last_merge_report)
        merge_report["mask_recovery"] = mask_recovery
        merge_report["language"] = language_report

        return {
            "image_name": image_name,
            "regions": merged_regions,
            "initial_masks": sam_masks,
            "edge_map": edge_map,
            "depth_map": depth_map,  # Add depth map to output
            "mask_recovery": mask_recovery,
            "language_report": language_report,
            "merge_report": merge_report,
        }

    def _resolve_language_device(self, device: str) -> str:
        if not str(device).startswith("cuda"):
            return device
        if not torch.cuda.is_available():
            print("Warning: CUDA requested for language model but unavailable. Falling back to CPU.")
            return "cpu"
        try:
            capability = torch.cuda.get_device_capability(torch.device(device))
            arch = f"sm_{capability[0]}{capability[1]}"
            if arch not in torch.cuda.get_arch_list():
                print(f"Warning: current PyTorch build does not support {arch}; using CPU for language model.")
                return "cpu"
        except Exception as exc:
            print(f"Warning: could not verify language CUDA support ({exc}); using CPU.")
            return "cpu"
        return device

    def _initialize_language_model(self) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("open_clip is required for --language-query. Install open_clip_torch.") from exc
        self.language_device = self._resolve_language_device(self.config.language_device)
        print(
            f"Loading language model: {self.config.language_model_name} "
            f"({self.config.language_pretrained}) on {self.language_device}"
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.language_model_name,
            pretrained=self.config.language_pretrained,
            device=self.language_device,
        )
        self.language_model = model.eval()
        self.language_preprocess = preprocess
        self.language_tokenizer = open_clip.get_tokenizer(self.config.language_model_name)

    def _compute_language_scores(
        self,
        image: np.ndarray,
        masks: Sequence[np.ndarray],
    ) -> Tuple[
        Optional[List[float]],
        Optional[List[Dict[str, float]]],
        Optional[List[Dict[str, float]]],
        Dict[str, object],
    ]:
        queries = self.config.language_queries or []
        if not queries:
            return None, None, None, {"enabled": False, "queries": []}
        if self.language_model is None:
            self._initialize_language_model()
        if not masks:
            return [], [], [], {
                "enabled": True,
                "queries": list(queries),
                "num_masks": 0,
                "judge": {"enabled": bool(self.config.use_language_judge), "groups": []},
            }

        pil_image = Image.fromarray(image).convert("RGB")
        crops = [self._make_language_crop(pil_image, mask) for mask in masks]
        raw_scores: List[float] = []
        query_scores_by_mask: List[List[float]] = []
        batch_size = max(1, int(self.config.language_batch_size))
        with torch.no_grad():
            tokens = self.language_tokenizer(list(queries)).to(self.language_device)
            text_features = self.language_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            for start in range(0, len(crops), batch_size):
                batch = crops[start : start + batch_size]
                tensor = torch.stack([self.language_preprocess(crop) for crop in batch], dim=0).to(self.language_device)
                image_features = self.language_model.encode_image(tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                scores = image_features @ text_features.T
                for row in scores.detach().cpu().numpy():
                    values = [float(v) for v in row.tolist()]
                    query_scores_by_mask.append(values)
                    raw_scores.append(float(max(values)))

        raw_by_query = np.asarray(query_scores_by_mask, dtype=np.float32)
        raw = np.asarray(raw_scores, dtype=np.float32)
        if raw.size > 0 and float(raw.max() - raw.min()) > 1e-6:
            normalized = ((raw - raw.min()) / (raw.max() - raw.min())).astype(np.float32)
        else:
            normalized = np.zeros_like(raw, dtype=np.float32)

        query_score_dicts: List[Dict[str, float]] = [dict() for _ in masks]
        group_score_dicts: List[Dict[str, float]] = [dict() for _ in masks]
        judge_enabled = bool(self.config.use_language_judge)
        min_group_masks = max(2, int(self.config.language_judge_min_group_masks))
        judge_groups: List[Dict[str, object]] = []

        for q_idx, query in enumerate(queries):
            query_raw = raw_by_query[:, q_idx]
            if query_raw.size > 0 and float(query_raw.max() - query_raw.min()) > 1e-6:
                query_norm = ((query_raw - query_raw.min()) / (query_raw.max() - query_raw.min())).astype(np.float32)
            else:
                query_norm = np.zeros_like(query_raw, dtype=np.float32)

            order = np.argsort(-query_norm)
            group_confidence = 0.0
            if judge_enabled and len(order) >= min_group_masks:
                top_values = np.clip(query_norm[order[:min_group_masks]], 0.0, 1.0)
                if float(top_values.min()) > 0.0:
                    group_confidence = float(np.exp(np.log(top_values + 1e-6).mean()))
            group_values = query_norm * group_confidence if judge_enabled else np.zeros_like(query_norm)
            soft_group_size = float(group_values.sum())
            strong_group_size = int((group_values >= float(self.config.language_judge_prior)).sum())
            oversegmentation_score = float(group_confidence * max(0.0, soft_group_size - 1.0))
            oversegmentation_candidate = bool(judge_enabled and strong_group_size >= min_group_masks)

            top_masks = []
            for idx in order[: min(10, len(order))]:
                mask = masks[int(idx)]
                group_score = float(group_values[int(idx)]) if judge_enabled else 0.0
                top_masks.append({
                    "mask_index": int(idx),
                    "raw_score": float(query_raw[int(idx)]),
                    "normalized_score": float(query_norm[int(idx)]),
                    "group_score": group_score,
                    "area": int(mask.sum()),
                    "bbox": list(_compute_bbox(mask)),
                })

            judge_groups.append({
                "query": str(query),
                "enabled": judge_enabled,
                "min_group_masks": int(min_group_masks),
                "group_confidence": float(group_confidence),
                "soft_group_size": float(soft_group_size),
                "strong_group_size": int(strong_group_size),
                "oversegmentation_score": float(oversegmentation_score),
                "oversegmentation_candidate": bool(oversegmentation_candidate),
                "raw_min": float(query_raw.min()) if query_raw.size else 0.0,
                "raw_max": float(query_raw.max()) if query_raw.size else 0.0,
                "top_masks": top_masks,
            })

            for idx in range(len(masks)):
                query_score = float(query_norm[idx])
                query_score_dicts[idx][str(query)] = query_score
                if judge_enabled:
                    group_score_dicts[idx][str(query)] = float(group_values[idx])

        top = []
        for idx in np.argsort(-normalized)[: min(10, len(normalized))]:
            mask = masks[int(idx)]
            top.append({
                "mask_index": int(idx),
                "raw_score": float(raw[int(idx)]),
                "normalized_score": float(normalized[int(idx)]),
                "area": int(mask.sum()),
                "bbox": list(_compute_bbox(mask)),
                "query_scores": {query: float(query_scores_by_mask[int(idx)][q_idx]) for q_idx, query in enumerate(queries)},
                "query_normalized_scores": dict(query_score_dicts[int(idx)]),
                "query_group_scores": dict(group_score_dicts[int(idx)]) if judge_enabled else {},
            })
        report = {
            "enabled": True,
            "queries": list(queries),
            "model": self.config.language_model_name,
            "pretrained": self.config.language_pretrained,
            "device": self.language_device,
            "score_prior": float(self.config.language_score_prior),
            "weight": float(self.config.language_weight),
            "num_masks": int(len(masks)),
            "raw_min": float(raw.min()) if raw.size else 0.0,
            "raw_max": float(raw.max()) if raw.size else 0.0,
            "top_masks": top,
            "judge": {
                "enabled": judge_enabled,
                "prior": float(self.config.language_judge_prior),
                "weight": float(self.config.language_judge_weight),
                "min_group_masks": int(min_group_masks),
                "groups": judge_groups,
            },
        }
        return (
            [float(v) for v in normalized.tolist()],
            query_score_dicts,
            group_score_dicts if judge_enabled else None,
            report,
        )

    def _make_language_crop(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        width, height = image.size
        x0, y0, x1, y1 = _compute_bbox(mask)
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        pad = int(round(max(bw, bh) * float(self.config.language_crop_pad)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(width - 1, x1 + pad)
        y1 = min(height - 1, y1 + pad)

        image_np = np.asarray(image)
        crop = image_np[y0 : y1 + 1, x0 : x1 + 1].copy()
        mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1]
        background = str(self.config.language_background).lower()
        if background == "image":
            return Image.fromarray(crop)
        if background == "black":
            bg = np.zeros_like(crop)
        elif background == "white":
            bg = np.full_like(crop, 255)
        elif background == "gray":
            bg = np.full_like(crop, 127)
        else:
            try:
                from PIL import ImageFilter
                blurred = image.filter(ImageFilter.GaussianBlur(radius=12))
                bg = np.asarray(blurred)[y0 : y1 + 1, x0 : x1 + 1].copy()
            except Exception:
                bg = crop.copy()
        bg[mask_crop] = crop[mask_crop]
        return Image.fromarray(bg)

    def _filter_recovered_masks_with_scorer(
        self,
        sam_masks: List[np.ndarray],
        feature_map: torch.Tensor,
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray],
        mask_recovery: Dict[str, object],
        image_shape: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], Dict[str, object]]:
        original_count = int(mask_recovery.get("input_masks", len(sam_masks)))
        recovered_count = max(0, len(sam_masks) - original_count)
        if recovered_count == 0:
            mask_recovery = dict(mask_recovery)
            mask_recovery["recovery_scorer"] = {
                "enabled": bool(self.config.use_recovery_scorer),
                "accepted": 0,
                "rejected": 0,
                "scorer": self.recovery_scorer.describe(),
                "candidates": [],
            }
            return sam_masks, mask_recovery

        original_masks = sam_masks[:original_count]
        candidate_masks = sam_masks[original_count:]
        original_regions = self.feature_lifter.build_regions(original_masks, feature_map, depth_map)
        candidate_regions = self.feature_lifter.build_regions(candidate_masks, feature_map, depth_map)

        accepted_masks = list(original_masks)
        candidate_reports = []
        accepted = 0
        rejected = 0
        for local_idx, candidate in enumerate(candidate_regions):
            evidence = self._build_recovery_evidence(
                candidate,
                original_regions,
                edge_map,
                depth_map,
                image_shape,
            )
            should_accept, score = self.recovery_scorer.should_accept(evidence)
            report_item = {
                "candidate_index": int(local_idx),
                "score": float(score),
                "accepted": bool(should_accept),
                "evidence": evidence.to_dict(),
                "bbox": list(candidate.bbox),
                "area": int(candidate.area),
            }
            candidate_reports.append(report_item)
            if should_accept:
                accepted_masks.append(candidate.mask)
                accepted += 1
            else:
                rejected += 1

        covered_after = _mask_union(accepted_masks, image_shape) if accepted_masks else np.zeros(image_shape, dtype=bool)
        image_area = max(1, int(image_shape[0]) * int(image_shape[1]))
        mask_recovery = dict(mask_recovery)
        mask_recovery["recovered_masks_before_scorer"] = int(recovered_count)
        mask_recovery["recovered_masks"] = int(accepted)
        mask_recovery["coverage_after"] = float(covered_after.sum() / image_area)
        mask_recovery["unassigned_pixels_after"] = int((~covered_after).sum())
        mask_recovery["recovery_scorer"] = {
            "enabled": True,
            "accepted": int(accepted),
            "rejected": int(rejected),
            "scorer": self.recovery_scorer.describe(),
            "candidates": candidate_reports,
        }
        return accepted_masks, mask_recovery

    def _build_recovery_evidence(
        self,
        candidate: MaskRegion,
        original_regions: List[MaskRegion],
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> RecoveryEvidence:
        height, width = image_shape
        candidate_dilated = binary_dilation(candidate.mask, iterations=1)
        boundary_union = np.zeros_like(candidate.mask, dtype=bool)
        sims = []
        depth_diffs = []
        contact_pixels = 0
        for region in original_regions:
            contact = np.logical_and(candidate_dilated, region.mask)
            if not contact.any():
                continue
            boundary_union |= contact
            contact_pixels += int(contact.sum())
            sims.append(self.merger._feature_similarity(candidate, region))
            if depth_map is not None:
                depth_diffs.append(abs(candidate.mean_depth - region.mean_depth))

        x0, y0, x1, y1 = candidate.bbox
        bbox_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        touches_border = x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1
        edge_strength = float(edge_map[boundary_union].mean()) if boundary_union.any() else 1.0
        return RecoveryEvidence(
            feature_similarity=float(max(sims)) if sims else 0.0,
            edge_strength=edge_strength,
            contact_ratio=float(contact_pixels / max(1, candidate.area)),
            area_ratio=float(candidate.area / max(1, height * width)),
            bbox_fill_ratio=float(candidate.area / bbox_area),
            depth_difference=float(min(depth_diffs)) if depth_diffs else 0.0,
            depth_available=bool(depth_diffs),
            has_neighbor=bool(sims),
            touches_border=bool(touches_border),
        )

    def precompute_dino_feature(self, image: np.ndarray, image_name: str) -> None:
        if not self.feature_cache_dir:
            raise ValueError("Feature cache directory is not configured.")
        if load_feature_from_cache(self.feature_cache_dir, image_name) is not None:
            return
        feature_map = self.feature_lifter.compute_feature_map(image)
        save_feature_to_cache(self.feature_cache_dir, image_name, feature_map, precision=self.dino_cache_precision)

    def _get_feature_map(self, image: np.ndarray, image_name: Optional[str]) -> torch.Tensor:
        if self.feature_cache_dir and image_name:
            cached = load_feature_from_cache(self.feature_cache_dir, image_name)
            if cached is not None:
                return cached
        feature_map = self.feature_lifter.compute_feature_map(image)
        if self.feature_cache_dir and image_name:
            save_feature_to_cache(self.feature_cache_dir, image_name, feature_map, precision=self.dino_cache_precision)
        return feature_map

    def _prepare_sam_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.sam_max_long_edge <= 0:
            return image, 1.0
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.sam_max_long_edge:
            return image, 1.0
        scale = self.sam_max_long_edge / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resized = Image.fromarray(image).resize(new_size, Image.BICUBIC)
        return np.array(resized), scale

    def _rescale_masks(
        self,
        masks: List[np.ndarray],
        target_shape: Tuple[int, int],
        scale: float,
    ) -> List[np.ndarray]:
        if not masks:
            return []
        if scale == 1.0:
            return masks
        target_h, target_w = target_shape
        resized = []
        for mask in masks:
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            upsampled = mask_img.resize((target_w, target_h), Image.NEAREST)
            resized.append(np.array(upsampled) > 0)
        return resized

    def _compute_depth_gradients(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute normalized depth gradients for boundary suppression."""
        sigma = max(0.0, self.config.depth_gradient_sigma)
        if cv2 is not None:
            depth_smooth = cv2.GaussianBlur(depth_map, (0, 0), sigma) if sigma > 0 else depth_map
            grad_x = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=3)
        else:
            depth_smooth = depth_map
            grad_y, grad_x = np.gradient(depth_smooth)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag -= grad_mag.min()
        denom = grad_mag.max()
        if denom < 1e-8:
            return np.zeros_like(grad_mag, dtype=np.float32)
        grad_norm = grad_mag / denom
        return grad_norm.astype(np.float32)

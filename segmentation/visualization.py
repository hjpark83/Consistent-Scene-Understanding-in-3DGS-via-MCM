from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .feature_field_pipeline import MaskRegion


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def _random_colors(num: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.0, 1.0, size=(num, 3))
    return colors.astype(np.float32)


def _stack_masks(regions: Sequence[MaskRegion]) -> np.ndarray:
    if not regions:
        raise ValueError("No regions to visualize.")
    return np.stack([region.mask for region in regions], axis=0)


def save_mask_overlay(
    image: np.ndarray,
    regions: Sequence[MaskRegion],
    out_path: Path,
    alpha: float = 0.45,
    seed: int = 0,
) -> None:
    if not regions:
        return

    masks = _stack_masks(regions)
    colors = _random_colors(len(regions), seed=seed)

    base = _ensure_uint8(image).astype(np.float32) / 255.0
    overlay = base.copy()

    for mask, color in zip(masks, colors):
        mask_idx = mask.astype(bool)
        overlay[mask_idx] = (1 - alpha) * overlay[mask_idx] + alpha * color

    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_img.save(out_path)


def save_label_map(
    regions: Sequence[MaskRegion],
    out_path: Path,
    seed: int = 0,
) -> None:
    if not regions:
        return
    masks = _stack_masks(regions)
    colors = (_random_colors(len(regions), seed=seed) * 255).astype(np.uint8)
    height, width = masks.shape[1:]
    label_img = np.zeros((height, width, 3), dtype=np.uint8)
    for mask, color in zip(masks, colors):
        label_img[mask] = color
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_img).save(out_path)


def save_edge_map(edge_map: np.ndarray, out_path: Path) -> None:
    if edge_map.size == 0:
        return
    edge = edge_map - edge_map.min()
    denom = max(edge.max(), 1e-6)
    edge_norm = (edge / denom * 255.0).astype(np.uint8)
    rgb_edge = np.stack([edge_norm] * 3, axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_edge).save(out_path)


def save_feature_field_visualizations(
    image: np.ndarray,
    regions: Sequence[MaskRegion],
    edge_map: np.ndarray,
    out_dir: Path,
    image_name: str,
    alpha: float = 0.5,
    seed: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if regions:
        save_mask_overlay(
            image=image,
            regions=regions,
            out_path=out_dir / f"{image_name}_overlay.png",
            alpha=alpha,
            seed=seed,
        )
        save_label_map(
            regions=regions,
            out_path=out_dir / f"{image_name}_labels.png",
            seed=seed,
        )
    if edge_map.size > 0:
        save_edge_map(edge_map, out_dir / f"{image_name}_edges.png")

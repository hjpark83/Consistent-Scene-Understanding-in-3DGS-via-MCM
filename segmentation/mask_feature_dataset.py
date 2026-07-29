from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch


def _build_label_map(masks: np.ndarray, areas: Optional[np.ndarray] = None) -> np.ndarray:
    if masks.size == 0:
        raise ValueError("Cannot build label map from empty mask stack.")

    num_masks, height, width = masks.shape
    label_map = np.full((height, width), fill_value=-1, dtype=np.int32)

    if areas is None or len(areas) != num_masks:
        order = range(num_masks)
    else:
        order = np.argsort(areas)[::-1]  # large regions first

    for idx in order:
        mask = masks[idx]
        label_map[mask] = idx

    return label_map


@dataclass
class MaskFeatureViewData:
    image_name: str
    height: int
    width: int
    mask_indices: torch.Tensor  # (H, W) int32, -1 for background
    features: torch.Tensor  # (N, D) float32
    areas: torch.Tensor  # (N,)
    bboxes: torch.Tensor  # (N, 4)
    edge_map: torch.Tensor  # (H, W)
    mask_weights: torch.Tensor  # (N,)
    source_ids: Optional[torch.Tensor] = None  # (N, K) or None
    metadata: Dict[str, object] = field(default_factory=dict)

    _device_cache: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    @property
    def feature_dim(self) -> int:
        return 0 if self.features.numel() == 0 else self.features.shape[1]

    def to_device(self, device: torch.device) -> Dict[str, torch.Tensor]:
        device_str = str(device)
        cache = self._device_cache.get(device_str)
        if cache is not None:
            return cache

        cache = {
            "mask_indices": self.mask_indices.to(device, non_blocking=True),
            "features": self.features.to(device, non_blocking=True),
            "areas": self.areas.to(device, non_blocking=True),
            "bboxes": self.bboxes.to(device, non_blocking=True),
            "mask_weights": self.mask_weights.to(device, non_blocking=True),
        }
        if self.source_ids is not None:
            cache["source_ids"] = self.source_ids.to(device, non_blocking=True)

        self._device_cache[device_str] = cache
        return cache

    @classmethod
    def from_npz(cls, path: Path) -> "MaskFeatureViewData":
        data = np.load(path, allow_pickle=True)

        masks = data["masks"].astype(bool)
        features = data["features"].astype(np.float32)
        areas = data["areas"].astype(np.int32)
        bboxes = data["bboxes"].astype(np.int32)
        edge_map = data["edge_map"].astype(np.float32)
        source_ids = data["source_ids"].astype(np.int32) if "source_ids" in data else None
        if "mask_weights" in data:
            mask_weights = data["mask_weights"].astype(np.float32)
            if mask_weights.shape[0] != features.shape[0]:
                raise ValueError(
                    f"mask_weights length {mask_weights.shape[0]} does not match masks {features.shape[0]} in {path}"
                )
        else:
            mask_weights = np.ones((features.shape[0],), dtype=np.float32)
        image_name = str(data["image_name"]) if "image_name" in data else path.stem

        if masks.size == 0:
            height = data["edge_map"].shape[0]
            width = data["edge_map"].shape[1]
            label_map = np.full((height, width), fill_value=-1, dtype=np.int32)
        else:
            _, height, width = masks.shape
            label_map = _build_label_map(masks, areas)

        return cls(
            image_name=image_name,
            height=height,
            width=width,
            mask_indices=torch.from_numpy(label_map.copy()),
            features=torch.from_numpy(features.copy()),
            areas=torch.from_numpy(areas.copy()),
            bboxes=torch.from_numpy(bboxes.copy()),
            edge_map=torch.from_numpy(edge_map.copy()),
            mask_weights=torch.from_numpy(mask_weights.copy()),
            source_ids=torch.from_numpy(source_ids.copy()) if source_ids is not None else None,
            metadata={"path": str(path)},
        )


def load_mask_feature_directory(directory: Path) -> Dict[str, MaskFeatureViewData]:
    views: Dict[str, MaskFeatureViewData] = {}
    npz_files: Iterable[Path] = sorted(directory.glob("*.npz"))

    for npz_path in npz_files:
        view = MaskFeatureViewData.from_npz(npz_path)
        views[view.image_name] = view

    return views

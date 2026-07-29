from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def _feature_path(cache_dir: Path, image_name: str) -> Path:
    return cache_dir / f"{image_name}.pt"


def load_feature_from_cache(cache_dir: Path, image_name: str) -> Optional[torch.Tensor]:
    path = _feature_path(cache_dir, image_name)
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    tensor = data.get("feature")
    if tensor is None:
        return None
    return tensor


def save_feature_to_cache(cache_dir: Path, image_name: str, feature: torch.Tensor, precision: str = "float32") -> None:
    path = _feature_path(cache_dir, image_name)
    if path.exists():
        return
    tensor = feature.clone().detach().cpu()
    if precision == "float16":
        tensor = tensor.half()
    elif precision == "bfloat16":
        tensor = tensor.bfloat16()
    torch.save({"feature": tensor}, path)

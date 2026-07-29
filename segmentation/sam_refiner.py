import math
import numpy as np
from typing import Any, Dict, List, Optional

import torch

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    _SAM_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - handled at runtime when SAM is missing
    SamAutomaticMaskGenerator = None
    sam_model_registry = None
    _SAM_IMPORT_ERROR = exc

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


class SAMMaskGenerator:

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        fallback_device: str = "cpu",
        generator_kwargs: Optional[Dict[str, Any]] = None,
        min_mask_area: int = 0,
    ) -> None:
        if SamAutomaticMaskGenerator is None or sam_model_registry is None:
            raise ImportError(
                "Segment Anything is required for SAMMaskGenerator. "
                "Install it via `pip install git+https://github.com/facebookresearch/segment-anything.git` "
            ) from _SAM_IMPORT_ERROR
        self.requested_device = device
        self.fallback_device = fallback_device
        self.device = self._resolve_device(device)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.device = self._move_model_to_device(sam, self.device)
        sam.eval() 

        default_kwargs: Dict[str, Any] = dict(
            points_per_side=20,
            pred_iou_thresh=0.82,
            stability_score_thresh=0.93,
            box_nms_thresh=0.2,
            min_mask_region_area=1024,
            output_mode="coco_rle",
        )
        if generator_kwargs:
            default_kwargs.update(generator_kwargs)

        auto_reference = default_kwargs.pop("auto_reference_long_edge", 2048)
        self.auto_reference_long_edge = auto_reference
        self.auto_tune = bool(default_kwargs.pop("auto_tune", True))

        self.sam_model = sam
        self.base_generator_kwargs = default_kwargs
        self.base_points_per_side = default_kwargs.get("points_per_side", 20)
        self.base_pred_iou = default_kwargs.get("pred_iou_thresh", 0.82)
        self.base_stability = default_kwargs.get("stability_score_thresh", 0.93)
        self.base_min_region = default_kwargs.get("min_mask_region_area", 1024)
        self.min_points_per_side = 8
        self.max_points_per_side = self.base_points_per_side
        self.min_mask_area = max(0, int(min_mask_area))

    def generate(self, image: np.ndarray) -> List[np.ndarray]:
        if image.dtype != np.uint8:
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image

        generator_kwargs = dict(self.base_generator_kwargs)
        if self.auto_tune:
            tuned_kwargs = self._auto_tune_kwargs(image_uint8.shape[0], image_uint8.shape[1])
            generator_kwargs.update(tuned_kwargs)

        # Disable gradients for inference
        with torch.no_grad():
            mask_generator = SamAutomaticMaskGenerator(model=self.sam_model, **generator_kwargs)
            raw_masks = mask_generator.generate(image_uint8)

        # Handle tuple output (some SAM versions return (masks, extra_info))
        if isinstance(raw_masks, tuple):
            raw_masks = raw_masks[0]

        masks: List[np.ndarray] = []
        for mask_dict in raw_masks:
            # Each mask_dict should be a dictionary with 'segmentation' key
            if not isinstance(mask_dict, dict):
                continue

            if "segmentation" not in mask_dict:
                continue

            mask = self._decode_segmentation(mask_dict["segmentation"])

            if mask.sum() < self.min_mask_area:
                continue
            masks.append(mask)

        # Clean up
        del raw_masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return masks

    def _decode_segmentation(self, segmentation: Any) -> np.ndarray:
        if isinstance(segmentation, np.ndarray):
            return segmentation.astype(bool)
        if torch.is_tensor(segmentation):
            return segmentation.detach().cpu().numpy().astype(bool)
        if isinstance(segmentation, dict):
            if mask_utils is None:
                raise ImportError(
                    "Install via `pip install pycocotools` or set generator_kwargs={'output_mode': 'binary_mask'}."
                )
            decoded = mask_utils.decode(segmentation)
            return decoded.astype(bool)
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def _move_model_to_device(self, model: torch.nn.Module, device: str) -> str:
        try:
            model.to(device)
            return device
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.fallback_device and device != self.fallback_device:
                fallback = self._resolve_device(self.fallback_device)
                print(
                    f"Failed to load SAM on {device} (OOM). "
                    f"Falling back to {fallback}."
                )
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                model.to(fallback)
                return fallback
            raise

    def _auto_tune_kwargs(self, height: int, width: int) -> Dict[str, Any]:
        long_edge = max(height, width)
        scale = max(1.0, long_edge / float(self.auto_reference_long_edge))
        sqrt_scale = math.sqrt(scale)

        points = int(round(self.base_points_per_side / sqrt_scale))
        points = int(np.clip(points, self.min_points_per_side, self.max_points_per_side))

        pred_iou = max(0.7, self.base_pred_iou - 0.04 * math.log(scale + 1.0, 2))
        stability = max(0.7, self.base_stability - 0.04 * math.log(scale + 1.0, 2))
        min_region = int(self.base_min_region * scale)

        return {
            "points_per_side": points,
            "pred_iou_thresh": pred_iou,
            "stability_score_thresh": stability,
            "min_mask_region_area": max(64, min_region),
        }

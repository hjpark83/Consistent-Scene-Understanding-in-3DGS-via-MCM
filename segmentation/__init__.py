from .edge_utils import CannyEdgeDetector, LoGEdgeDetector
from .mask_feature_dataset import MaskFeatureViewData, load_feature_field_directory
from .visualization import save_feature_field_visualizations
from .dino_cache import load_feature_from_cache, save_feature_to_cache

__all__ = [
    "MaskFeatureConfig",
    "MaskFeaturePipeline",
    "SAMMaskGenerator",
    "CannyEdgeDetector",
    "LoGEdgeDetector",
    "MaskFeatureViewData",
    "load_feature_field_directory",
    "save_feature_field_visualizations",
    "load_feature_from_cache",
    "save_feature_to_cache",
]


def __getattr__(name):
    if name in {"MaskFeatureConfig", "MaskFeaturePipeline"}:
        from .mask_feature_pipeline import MaskFeatureConfig, MaskFeaturePipeline

        return {"MaskFeatureConfig": MaskFeatureConfig, "MaskFeaturePipeline": MaskFeaturePipeline}[name]
    if name == "SAMMaskGenerator":
        from .sam_refiner import SAMMaskGenerator

        return SAMMaskGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#!/usr/bin/env python3

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


# (yaml_section, yaml_key) -> shell variable name. Scene configs are nested;
# `dataset` is the only top-level key and is handled separately in main().
NESTED_KEY_MAP: Iterable[Tuple[Tuple[str, str], str]] = (
    (("runtime", "gpu"), "GPU_ID"),
    (("runtime", "iterations"), "ITERATIONS"),
    (("runtime", "sam_checkpoint"), "SAM_CHECKPOINT"),
    (("runtime", "depth_device"), "DEPTH_DEVICE"),
    (("runtime", "resolution"), "TRAIN_RESOLUTION"),
    (("runtime", "max_init_points"), "MAX_INIT_POINTS"),
    (("paths", "data"), "DATA_DIR"),
    (("paths", "output"), "OUTPUT_BASE"),
    (("stages", "feature_field"), "RUN_FEATURE_FIELD"),
    (("stages", "assign_mask_ids"), "RUN_MASK_ID_ASSIGN"),
    (("mcm", "feature_weight"), "MCM_FEATURE_WEIGHT"),
    (("mcm", "depth_weight"), "MCM_DEPTH_WEIGHT"),
    (("mcm", "edge_penalty"), "MCM_EDGE_PENALTY"),
    (("mcm", "depth_boundary_weight"), "MCM_DEPTH_BOUNDARY_WEIGHT"),
    (("mcm", "depth_diff_threshold"), "MCM_DEPTH_DIFF_THRESHOLD"),
    (("mcm", "depth_affinity_sigma2"), "MCM_DEPTH_AFFINITY_SIGMA2"),
    (("mcm", "merge_score_threshold"), "MCM_MERGE_SCORE_THRESHOLD"),
    (("mcm", "max_merge_iterations"), "MCM_MAX_MERGE_ITERATIONS"),
    (("mcm", "containment_ratio_threshold"), "MCM_CONTAINMENT_RATIO"),
    (("mcm", "containment_feature_threshold"), "MCM_CONTAINMENT_FEATURE"),
    (("cross_view", "threshold"), "MASK_MATCH_THRESHOLD"),
    (("cross_view", "max_view_gap"), "MASK_MATCH_MAX_VIEW_GAP"),
    (("cross_view", "topk_per_mask"), "MASK_MATCH_TOPK"),
    (("lifting", "zbuffer_abs_tolerance"), "ZBUFFER_ABS_TOLERANCE"),
    (("lifting", "zbuffer_rel_tolerance"), "ZBUFFER_REL_TOLERANCE"),
)


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def emit_assignment(name: str, value: Any) -> None:
    print(f"{name}={shlex.quote(normalize_value(value))}")


def read_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML pipeline configs") from exc
            payload = yaml.safe_load(f)
        else:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("pipeline config must be a mapping/object")
    return payload


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


BASE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pipeline" / "base.yaml"


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: load_pipeline_config.py [config.yaml|config.json]")

    config = read_config(BASE_CONFIG_PATH) if BASE_CONFIG_PATH.exists() else {}

    if len(sys.argv) == 2:
        scene_path = Path(sys.argv[1]).expanduser()
        config = deep_merge(config, read_config(scene_path))

    if "dataset" in config:
        emit_assignment("DATASET_NAME", config["dataset"])

    for key_path, env_name in NESTED_KEY_MAP:
        section = config.get(key_path[0], {})
        if isinstance(section, dict) and key_path[1] in section:
            emit_assignment(env_name, section[key_path[1]])


if __name__ == "__main__":
    main()

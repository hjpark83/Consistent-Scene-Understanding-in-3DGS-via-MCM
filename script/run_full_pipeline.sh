#!/bin/bash

################################################################################
# Full Pipeline: Depth-Aware Feature-Field 3D Gaussian Splatting
# This script runs the complete pipeline from COLMAP to final rendering
################################################################################

set -e  # Exit on error

section() {
    echo ""
    echo "[MCM] $1"
}

info() {
    echo "  $1"
}

# ============================================================================
# Configuration
#
# Usage:
#   bash script/run_full_pipeline.sh --config config/pipeline/base.yaml <dataset> [data_dir] [output_base]
# ============================================================================

PIPELINE_CONFIG=""
if [ "${1:-}" = "--config" ]; then
    if [ -z "${2:-}" ]; then
        echo "Error: --config requires a YAML/JSON config path"
        exit 1
    fi
    PIPELINE_CONFIG=$2
    shift 2
fi

# All hyperparameter values come from config/pipeline/base.yaml, with the
# scene config (--config) deep-merged on top. See load_pipeline_config.py.
# A GPU_ID set in the environment before the script runs wins over the YAML.
ENV_GPU_ID=${GPU_ID:-}
if [ -n "${PIPELINE_CONFIG}" ]; then
    eval "$(python script/load_pipeline_config.py "${PIPELINE_CONFIG}")"
else
    eval "$(python script/load_pipeline_config.py)"
fi
GPU_ID=${ENV_GPU_ID:-${GPU_ID:-1}}

# Positional args: <dataset> [data_dir] [output_base]. 
DATASET_NAME=${1:-${DATASET_NAME:-"table"}}
DATA_DIR=${2:-${DATA_DIR:-"data/${DATASET_NAME}"}}
OUTPUT_BASE=${3:-${OUTPUT_BASE:-"output/${DATASET_NAME}"}}
ASSIGN_EXTRA_ARGS=("${@:4}")

# CUDA_DEVICE_ORDER has no YAML key; keep its default here.
CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-"PCI_BUS_ID"}

TRAIN_RESOLUTION_ARGS=""
if [ -n "${TRAIN_RESOLUTION}" ]; then
    TRAIN_RESOLUTION_ARGS="-r ${TRAIN_RESOLUTION}"
fi

export CUDA_DEVICE_ORDER
export MAX_INIT_POINTS
export MCM_FEATURE_WEIGHT
export MCM_DEPTH_WEIGHT
export MCM_EDGE_PENALTY
export MCM_DEPTH_BOUNDARY_WEIGHT
export MCM_DEPTH_DIFF_THRESHOLD
export MCM_DEPTH_AFFINITY_SIGMA2
export MCM_MERGE_SCORE_THRESHOLD
export MCM_MAX_MERGE_ITERATIONS
export MCM_CONTAINMENT_RATIO
export MCM_CONTAINMENT_FEATURE
export MASK_MATCH_THRESHOLD
export MASK_MATCH_MAX_VIEW_GAP
export MASK_MATCH_TOPK

# Derived paths
IMAGE_DIR="${DATA_DIR}/images"
SPARSE_DIR="${DATA_DIR}/sparse"
DEPTH_CACHE_DIR="cache/depth_maps/${DATASET_NAME}"
DINO_CACHE_DIR="cache/dino_features/${DATASET_NAME}"
FEATURE_FIELD_DIR="${OUTPUT_BASE}/feature_field"
MODEL_DIR="${OUTPUT_BASE}/depth_aware"
RENDER_ROOT=${RENDER_ROOT:-"${OUTPUT_BASE}/render"}
TRAIN_CONFIG_FILE=${PIPELINE_CONFIG}

section "Pipeline start"
info "data=${DATA_DIR}"
info "output=${OUTPUT_BASE}"
info "iterations=${ITERATIONS}"

CUDA_VISIBLE_DEVICES=${GPU_ID}

# ============================================================================
# Step 1: Verify COLMAP data
# ============================================================================

section "1/9 Verify COLMAP data"

if [ ! -d "${SPARSE_DIR}" ]; then
    echo "❌ Error: COLMAP sparse directory not found: ${SPARSE_DIR}"
    exit 1
fi

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "❌ Error: Images directory not found: ${IMAGE_DIR}"
    exit 1
fi

NUM_IMAGES=$(ls ${IMAGE_DIR}/*.{jpg,png,JPG,PNG} 2>/dev/null | wc -l)
info "images=${NUM_IMAGES}"

# ============================================================================
# Step 2: Precompute depth maps
# ============================================================================

section "2/9 Precompute depth maps"

if [ -d "${DEPTH_CACHE_DIR}" ] && [ "$(ls -A ${DEPTH_CACHE_DIR}/*.npy 2>/dev/null | wc -l)" -gt 0 ]; then
    info "depth cache exists: ${DEPTH_CACHE_DIR}"
else
    CUDA_VISIBLE_DEVICES=${RESOLVED_GPU_ID} python script/precompute_depth_maps.py \
        --image-dir ${IMAGE_DIR} \
        --output-dir ${DEPTH_CACHE_DIR} \
        --device ${DEPTH_DEVICE}
fi

info "depth=${DEPTH_CACHE_DIR}"

# ============================================================================
# Step 3: Feature-field segmentation
# ============================================================================

section "3/9 MCM feature-field segmentation"

FEATURE_FIELD_COUNT=$(ls -A ${FEATURE_FIELD_DIR}/*.npz 2>/dev/null | wc -l)

if [ "${RUN_FEATURE_FIELD}" = "0" ]; then
    info "skipped (RUN_FEATURE_FIELD=0), files=${FEATURE_FIELD_COUNT}/${NUM_IMAGES}"
elif [ -d "${FEATURE_FIELD_DIR}" ] && [ "${FEATURE_FIELD_COUNT}" -ge "${NUM_IMAGES}" ]; then
    info "feature field cache exists: ${FEATURE_FIELD_COUNT}/${NUM_IMAGES}"
else
    CUDA_VISIBLE_DEVICES=${RESOLVED_GPU_ID} python script/run_feature_field_segmentation.py \
        --image-dir ${IMAGE_DIR} \
        --output-dir ${FEATURE_FIELD_DIR} \
        --sam-checkpoint ${SAM_CHECKPOINT} \
        --dino-cache-dir ${DINO_CACHE_DIR} \
        --depth-cache-dir ${DEPTH_CACHE_DIR}
fi

info "feature_field=${FEATURE_FIELD_DIR}"

# ============================================================================
# Step 4: Check pipeline config
# ============================================================================

section "4/9 Check dataset config"

if [ -z "${TRAIN_CONFIG_FILE}" ]; then
    echo "Error: run_full_pipeline.sh requires --config config/pipeline/base.yaml or a scene config"
    exit 1
elif [ ! -f "${TRAIN_CONFIG_FILE}" ]; then
    echo "Error: pipeline config not found: ${TRAIN_CONFIG_FILE}"
    exit 1
else
    info "config=${TRAIN_CONFIG_FILE}"
fi

# ============================================================================
# Step 5: Train 3D Gaussian Splatting with feature-field
# ============================================================================

section "5/9 Train 3DGS"

POINT_CLOUD_FILE="${MODEL_DIR}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
CHECKPOINT_FILE="${MODEL_DIR}/chkpnt_auto_${ITERATIONS}.pth"
TRAIN_CACHE_EXISTS=0
if [ -f "${POINT_CLOUD_FILE}" ] || [ -f "${CHECKPOINT_FILE}" ]; then
    TRAIN_CACHE_EXISTS=1
fi

if [ "${FORCE_TRAIN}" = "1" ] || [ "${TRAIN_CACHE_EXISTS}" != "1" ]; then
    CUDA_VISIBLE_DEVICES=${RESOLVED_GPU_ID} python train.py \
        -s ${DATA_DIR} \
        -m ${MODEL_DIR} \
        --iterations ${ITERATIONS} \
        ${TRAIN_RESOLUTION_ARGS} \
        --feature_field_dir ${FEATURE_FIELD_DIR} \
        --config_file ${TRAIN_CONFIG_FILE}
else
    info "training cache exists"
    info "point_cloud=${POINT_CLOUD_FILE}"
    info "set FORCE_TRAIN=1 to retrain"
fi

info "model=${MODEL_DIR}"

# ============================================================================
# Step 6: Assign 3D mask IDs from refined 2D masks
# ============================================================================

section "6/9 Assign 3D mask IDs"

if [ "${RUN_MASK_ID_ASSIGN}" = "1" ]; then
    if [ "${#ASSIGN_EXTRA_ARGS[@]}" -gt 0 ]; then
        info "assign_extra_args=${ASSIGN_EXTRA_ARGS[*]}"
    fi
    CUDA_VISIBLE_DEVICES=${RESOLVED_GPU_ID} python script/assign_mask_ids_to_point_cloud.py \
        -s ${DATA_DIR} \
        -m ${MODEL_DIR} \
        ${TRAIN_RESOLUTION_ARGS} \
        --iteration ${ITERATIONS} \
        --feature_field_dir ${FEATURE_FIELD_DIR} \
        --feature_field_matching_threshold ${MASK_MATCH_THRESHOLD} \
        --matching_max_view_gap ${MASK_MATCH_MAX_VIEW_GAP} \
        --matching_topk_per_mask ${MASK_MATCH_TOPK} \
        --zbuffer_abs_tolerance ${ZBUFFER_ABS_TOLERANCE} \
        --zbuffer_rel_tolerance ${ZBUFFER_REL_TOLERANCE} \
        --overwrite_point_cloud \
        --no_smooth_features_by_mask \
        "${ASSIGN_EXTRA_ARGS[@]}"
else
    info "skipped (RUN_MASK_ID_ASSIGN=${RUN_MASK_ID_ASSIGN})"
fi

info "mask_id_ply=${POINT_CLOUD_FILE}"

# ============================================================================
# Step 7: Render with mask-refinement visualizations
# ============================================================================

section "7/9 Render mask refinement results"

CUDA_VISIBLE_DEVICES=${RESOLVED_GPU_ID} python render/mask_refinement.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    ${TRAIN_RESOLUTION_ARGS} \
    --iteration ${ITERATIONS} \
    --render_root ${RENDER_ROOT} \
    --skip_test

info "mask_refinement=${RENDER_ROOT}/mask_refinement/train/ours_${ITERATIONS}"

# ============================================================================
# Step 8: Visualize 3D mask IDs
# ============================================================================

section "8/9 Render global mask IDs"

CUDA_VISIBLE_DEVICES=${RESOLVED_GPU_ID} python render/object_editing.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    ${TRAIN_RESOLUTION_ARGS} \
    --iteration ${ITERATIONS} \
    --feature_field_dir ${FEATURE_FIELD_DIR} \
    --feature_field_matching_threshold ${MASK_MATCH_THRESHOLD} \
    --feature_field_matching_max_view_gap ${MASK_MATCH_MAX_VIEW_GAP} \
    --feature_field_matching_topk_per_mask ${MASK_MATCH_TOPK} \
    --render_root ${RENDER_ROOT} \
    --global_id_source mapping \
    --mode visualize

info "global_masks=${RENDER_ROOT}/global_mask_ids/ours_${ITERATIONS}"

# ============================================================================
# Step 9: Optionally render PCA Gaussian feature field
# ============================================================================

section "9/9 Feature outputs"

# View-consistent feature preview: fixed global PCA basis (shared across views)
# + global mask IDs colored by a fixed per-object palette, so each object keeps
# the same false color in every view. Step 8 must have produced global_mask_ids.
CUDA_VISIBLE_DEVICES=${RESOLVED_GPU_ID} python render/features.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    ${TRAIN_RESOLUTION_ARGS} \
    --iteration ${ITERATIONS} \
    --projection pca \
    --render_root ${RENDER_ROOT} \
    --mask_guidance_source global \
    --object_palette_guidance \
    --skip_test

python script/export_feature_viewer_model.py \
    -m ${MODEL_DIR} \
    --iteration ${ITERATIONS} \
    --output_model ${RENDER_ROOT}/feature_viewer_model \
    --prototype_blend 0.45

section "Complete"

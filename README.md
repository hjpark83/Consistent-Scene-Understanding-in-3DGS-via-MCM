# Consistent Scene Understanding in 3D Gaussian Splatting via Multi Cue Mask Refinement (ICPR 2026)

> **Hyunjoon Park**, and Donghyeon Cho

Official repository for our ICPR 2026 paper on Consistent Scene Understanding in reconstructed 3D space using Gaussian Splatting and 2D Segmentation Model.

### :triangular_flag_on_post: Updates
* **2026.4.1**: 🎉Paper accepted at ICPR 2026🎉
* **2026.05.14**: We released the [Environment Setting Notes](./docs/env_setting.md).
* **2026.07.29**: Update [README.md](./README.md) which includes commands to run the pipeline.

## Pipeline
<img width="1140" height="583" alt="Image" src="https://github.com/user-attachments/assets/60f62584-db15-4c0a-b9e3-73a22071aefa" />

## ⚙️ Env Setting

You can refer to the [env_setting.md](./docs/env_setting.md) file to build environment for our work. 

## 🚀 Process

```python
  SCENE=dataset_name
  DATA_DIR=data/${SCENE}
  OUTPUT_DIR=output/${SCENE}
  MODEL_DIR=${OUTPUT_DIR}/depth_aware
  FEATURE_FIELD_DIR=${OUTPUT_DIR}/feature_field
  RENDER_ROOT=${OUTPUT_DIR}/render
  DEPTH_CACHE_DIR=cache/depth_maps/${SCENE}
  DINO_CACHE_DIR=cache/dino_features/${SCENE}
  ITER=30000
  GPU=1
  SAM_CHECKPOINT=sam_vit_h_4b8939.pth
  CONFIG=config/pipeline/base.yaml
```

#### Full-pipeline

```bash
GPU_ID=${GPU} bash script/run_full_pipeline.sh \
    --config ${CONFIG} \
    ${SCENE} ${DATA_DIR} ${OUTPUT_DIR}
```

<details>
<summary>Commands</summary>
<div markdown="1">

### 1. Precompute Depth Maps

```bash
CUDA_VISIBLE_DEVICES=${GPU} python script/precompute_depth_maps.py \
    --image-dir ${DATA_DIR}/images \
    --output-dir ${DEPTH_CACHE_DIR} \
    --device cuda
```

### 2. MCM Feature-Field Segmentation

```bash
CUDA_VISIBLE_DEVICES=${GPU} python script/run_feature_field_segmentation.py \
    --image-dir ${DATA_DIR}/images \
    --output-dir ${FEATURE_FIELD_DIR} \
    --sam-checkpoint ${SAM_CHECKPOINT} \
    --dino-cache-dir ${DINO_CACHE_DIR} \
    --depth-cache-dir ${DEPTH_CACHE_DIR}
```

### 3. Train 3DGS With Feature Field

```bash
CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    --iterations ${ITER} \
    --feature_field_dir ${FEATURE_FIELD_DIR} \
    --config_file ${CONFIG}
```

### 4. Assign Global Mask IDs To Gaussians

```
CUDA_VISIBLE_DEVICES=${GPU} python script/assign_mask_ids_to_point_cloud.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    --iteration ${ITER} \
    --feature_field_dir ${FEATURE_FIELD_DIR} \
    --feature_field_matching_threshold 0.7 \
    --matching_max_view_gap 10 \
    --matching_topk_per_mask 3 \
    --zbuffer_abs_tolerance 0.02 \
    --zbuffer_rel_tolerance 0.03 \
    --overwrite_point_cloud \
    --no_smooth_features_by_mask
```

> Optional multi-view refinement:

```bash
--use_multiview_refinement
```

### 5. Render Mask Refinement Visualization

```bash
CUDA_VISIBLE_DEVICES=${GPU} python render/mask_refinement.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    --iteration ${ITER} \
    --render_root ${RENDER_ROOT} \
    --skip_test
```

### 6. Render Global Mask IDs

```bash
CUDA_VISIBLE_DEVICES=${GPU} python render/object_editing.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    --iteration ${ITER} \
    --feature_field_dir ${FEATURE_FIELD_DIR} \
    --feature_field_matching_threshold 0.7 \
    --feature_field_matching_max_view_gap 10 \
    --feature_field_matching_topk_per_mask 3 \
    --render_root ${RENDER_ROOT} \
    --global_id_source mapping \
    --mode visualize
```

### 7. Render PCA Feature Field

```bash
CUDA_VISIBLE_DEVICES=${GPU} python render/features.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    --iteration ${ITER} \
    --projection pca \
    --render_root ${RENDER_ROOT} \
    --mask_guidance_source global \
    --object_palette_guidance \
    --skip_test
```

### 8. Export Feature Viewer Model

```bash
python script/export_feature_viewer_model.py \
    -m ${MODEL_DIR} \
    --iteration ${ITER} \
    --output_model ${RENDER_ROOT}/feature_viewer_model \
    --prototype_blend 0.45
```

</div>
</details>

## 📚 Citation

```bibtex
```

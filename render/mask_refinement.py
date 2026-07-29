# Copyright (C) 2023, Gaussian-Grouping
# Modified for Feature-Field visualization

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
from segmentation.feature_field_dataset import build_label_map


def default_render_root(model_path):
    return os.path.join(os.path.dirname(os.path.normpath(model_path)), "render")


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    if id == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Use HSL color space for better color distribution
    h = (id * 0.618033988749895) % 1.0  # Golden ratio
    s = 0.5 + (id % 3) * 0.15
    l = 0.4 + (id % 4) * 0.1

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)


def visualize_regions(region_map):
    H, W = region_map.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)

    unique_ids = np.unique(region_map)
    for region_id in unique_ids:
        rgb_mask[region_map == region_id] = id2rgb(int(region_id))

    return rgb_mask


def create_mask_comparison(initial_mask, refined_mask, num_initial, num_refined):
    H, W = initial_mask.shape[:2]

    # Create divider line (white, 4 pixels wide)
    divider_width = 4
    divider = np.ones((H, divider_width, 3), dtype=np.uint8) * 255

    # Add text labels
    initial_with_text = initial_mask.copy()
    refined_with_text = refined_mask.copy()

    # Add text to images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Initial mask text
    text_initial = f"Initial SAM: {num_initial} regions"
    text_size = cv2.getTextSize(text_initial, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 30

    # Add black background for text
    cv2.rectangle(initial_with_text,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 0, 0), -1)
    cv2.putText(initial_with_text, text_initial, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness)

    # Refined mask text
    text_refined = f"Refined: {num_refined} regions"
    reduction = num_initial - num_refined
    reduction_pct = (reduction / num_initial * 100) if num_initial > 0 else 0
    text_refined2 = f"({reduction} merged, -{reduction_pct:.1f}%)"

    text_size = cv2.getTextSize(text_refined, font, font_scale, thickness)[0]
    cv2.rectangle(refined_with_text,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 0, 0), -1)
    cv2.putText(refined_with_text, text_refined, (text_x, text_y),
                font, font_scale, (0, 255, 0), thickness)

    # Add second line of text
    text_size2 = cv2.getTextSize(text_refined2, font, font_scale * 0.7, thickness - 1)[0]
    text_y2 = text_y + 30
    cv2.rectangle(refined_with_text,
                  (text_x - 5, text_y2 - text_size2[1] - 5),
                  (text_x + text_size2[0] + 5, text_y2 + 5),
                  (0, 0, 0), -1)
    cv2.putText(refined_with_text, text_refined2, (text_x, text_y2),
                font, font_scale * 0.7, (0, 255, 0), thickness - 1)

    # Concatenate: Initial | Divider | Refined
    comparison = np.hstack([initial_with_text, divider, refined_with_text])

    return comparison


def resize_to_hw(image, height, width, interpolation=cv2.INTER_NEAREST):
    if image.shape[0] == height and image.shape[1] == width:
        return image
    return cv2.resize(image, (width, height), interpolation=interpolation)


def load_feature_field_data(feature_field_dir, image_name):
    base_name = os.path.splitext(image_name)[0]

    # Try to find matching file
    possible_names = [
        f"{base_name}.npz",
        f"frame_{base_name}.npz",
    ]

    # Try numeric conversion
    if base_name.isdigit():
        possible_names.append(f"frame_{int(base_name):05d}.npz")

    mask_path = None
    for name in possible_names:
        test_path = os.path.join(feature_field_dir, name)
        if os.path.exists(test_path):
            mask_path = test_path
            break

    if mask_path is None:
        return None

    try:
        data = np.load(mask_path, allow_pickle=True)

        result = {}

        # 1. Refined masks (after hierarchical merging)
        if 'masks' in data:
            masks = data['masks']  # (N_refined, H, W) boolean
            N, H, W = masks.shape

            # Create region ID map
            region_map = build_label_map(masks) + 1

            result['refined_masks'] = visualize_regions(region_map)
            result['num_refined_regions'] = N

        # 2. Initial SAM masks. Prefer the true pre-merge masks saved by the MCM
        # pipeline; source_ids alone cannot reconstruct pre-merge geometry.
        if 'initial_masks' in data:
            initial_masks = data['initial_masks'].astype(bool)
            N_initial, H, W = initial_masks.shape
            initial_map = build_label_map(initial_masks) + 1

            result['initial_masks'] = visualize_regions(initial_map)
            result['num_initial_regions'] = N_initial

        elif 'source_ids' in data and 'masks' in data:
            source_ids = data['source_ids']  # (N_refined, max_sources)
            masks = data['masks']

            # Reconstruct initial SAM regions by assigning each source_id
            max_initial_id = source_ids.max()
            initial_map = np.zeros((H, W), dtype=np.int32)

            # Create mapping: for each refined region, assign all its source regions
            for refined_idx in range(len(masks)):
                # Get source IDs for this refined region
                sources = source_ids[refined_idx]
                sources = sources[sources >= 0]  # Filter out -1 padding

                # For visualization: assign each source a unique ID
                for src_id in sources:
                    if src_id >= 0:
                        # Assign this source ID to pixels in refined mask
                        initial_map[masks[refined_idx]] = src_id

            result['initial_masks'] = visualize_regions(initial_map)
            result['num_initial_regions'] = int(max_initial_id) + 1

        # 5. Depth map visualization
        if 'depth_map' in data:
            depth_map = data['depth_map']  # (H, W) float

            # Normalize to [0, 255]
            depth_norm = depth_map.copy()
            depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
            depth_viz = (depth_norm * 255).astype(np.uint8)

            # Paper-style relative-depth visualization.
            result['depth_map'] = cv2.applyColorMap(depth_viz, cv2.COLORMAP_PLASMA)

        # 3. Edge map (LoG)
        if 'edge_map' in data:
            edge_map = data['edge_map']  # (H, W) float

            # Normalize to [0, 255]
            edge_map_norm = edge_map.copy()
            edge_map_norm = (edge_map_norm - edge_map_norm.min()) / (edge_map_norm.max() - edge_map_norm.min() + 1e-8)
            edge_map_viz = (edge_map_norm * 255).astype(np.uint8)

            # Paper figure uses grayscale structural LoG edges.
            result['edge_map'] = cv2.cvtColor(edge_map_viz, cv2.COLOR_GRAY2RGB)

        # 4. DINO features visualization (PCA per region)
        if 'features' in data and 'masks' in data:
            features = data['features']  # (N, 384)
            masks = data['masks']

            if features.shape[0] > 0:
                # Apply PCA to reduce 384D -> 3D
                pca = PCA(n_components=3)
                features_reduced = pca.fit_transform(features)  # (N, 3)

                # Normalize to [0, 1]
                features_reduced = (features_reduced - features_reduced.min(axis=0)) / \
                                  (features_reduced.max(axis=0) - features_reduced.min(axis=0) + 1e-8)

                # Create feature visualization
                dino_viz = np.zeros((H, W, 3), dtype=np.float32)
                for i in range(len(masks)):
                    dino_viz[masks[i]] = features_reduced[i]

                result['dino_features'] = (dino_viz * 255).astype(np.uint8)

        return result

    except Exception as e:
        print(f"Failed to load feature-field data for {image_name}: {e}")
        return None


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, feature_field_dir=None, render_root=None):
    render_root = render_root or default_render_root(model_path)
    base_path = os.path.join(render_root, "mask_refinement", name, f"ours_{iteration}")
    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "gt")
    refined_mask_path = os.path.join(base_path, "refined_masks")
    initial_mask_path = os.path.join(base_path, "initial_sam_masks")
    comparison_path = os.path.join(base_path, "mask_comparison")
    edge_map_path = os.path.join(base_path, "edge_maps")
    dino_feature_path = os.path.join(base_path, "dino_features")
    depth_map_path = os.path.join(base_path, "depth_maps")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(refined_mask_path, exist_ok=True)
    makedirs(initial_mask_path, exist_ok=True)
    makedirs(comparison_path, exist_ok=True)
    makedirs(edge_map_path, exist_ok=True)
    makedirs(dino_feature_path, exist_ok=True)
    makedirs(depth_map_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 1. Render RGB
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :].detach().cpu()
        target_h, target_w = int(gt.shape[1]), int(gt.shape[2])

        # Save RGB and GT
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))

        # 2. Load and save feature-field visualizations
        if feature_field_dir is not None:
            ff_data = load_feature_field_data(feature_field_dir, view.image_name)

            if ff_data is not None:
                initial_mask_img = None
                refined_mask_img = None

                if 'refined_masks' in ff_data:
                    refined_mask_img = resize_to_hw(ff_data['refined_masks'], target_h, target_w, cv2.INTER_NEAREST)
                    Image.fromarray(refined_mask_img).save(
                        os.path.join(refined_mask_path, f'{idx:05d}.png'))

                if 'initial_masks' in ff_data:
                    initial_mask_img = resize_to_hw(ff_data['initial_masks'], target_h, target_w, cv2.INTER_NEAREST)
                    Image.fromarray(initial_mask_img).save(
                        os.path.join(initial_mask_path, f'{idx:05d}.png'))

                # Create side-by-side comparison: Initial | Refined
                if initial_mask_img is not None and refined_mask_img is not None:
                    comparison_img = create_mask_comparison(
                        initial_mask_img, refined_mask_img,
                        ff_data.get('num_initial_regions', 0),
                        ff_data.get('num_refined_regions', 0)
                    )
                    Image.fromarray(comparison_img).save(
                        os.path.join(comparison_path, f'{idx:05d}.png'))

                if 'edge_map' in ff_data:
                    edge_img = resize_to_hw(ff_data['edge_map'], target_h, target_w, cv2.INTER_LINEAR)
                    Image.fromarray(edge_img).save(
                        os.path.join(edge_map_path, f'{idx:05d}.png'))

                if 'dino_features' in ff_data:
                    dino_img = resize_to_hw(ff_data['dino_features'], target_h, target_w, cv2.INTER_NEAREST)
                    Image.fromarray(dino_img).save(
                        os.path.join(dino_feature_path, f'{idx:05d}.png'))

                if 'depth_map' in ff_data:
                    depth_img = resize_to_hw(ff_data['depth_map'], target_h, target_w, cv2.INTER_LINEAR)
                    Image.fromarray(depth_img).save(
                        os.path.join(depth_map_path, f'{idx:05d}.png'))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, render_root: str = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_root = render_root or default_render_root(dataset.model_path)
        os.makedirs(render_root, exist_ok=True)
        print(f"Render root: {render_root}")

        # Get feature-field directory
        feature_field_dir = getattr(dataset, 'feature_field_dir', None)
        if feature_field_dir and os.path.exists(feature_field_dir):
            print(f"Using feature-field masks from: {feature_field_dir}")
        else:
            print("No feature-field directory found")
            feature_field_dir = None

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                      gaussians, pipeline, background, feature_field_dir, render_root)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                      gaussians, pipeline, background, feature_field_dir, render_root)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script with feature-field visualization")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_root", type=str, default=None,
                        help="Output root for render artifacts. Default: dirname(model_path)/render")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(lp.extract(args), args.iteration, pp.extract(args), args.skip_train, args.skip_test, args.render_root)

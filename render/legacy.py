# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

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

def feature_to_rgb(features):
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array


def visualize_dino_features_simple(gaussians):
    """
    Simple PCA visualization of DINO features.
    Just reduces all DINO features to 3D RGB using PCA.
    Returns: (N, 3) numpy array of RGB values for each Gaussian
    """
    if not hasattr(gaussians, '_dino_features') or gaussians._dino_features is None:
        return None

    # Get DINO features
    dino_features = gaussians._dino_features.cpu().numpy()  # (N, 384)

    # Apply PCA to reduce 384D -> 3D
    pca = PCA(n_components=3)
    dino_reduced = pca.fit_transform(dino_features)  # (N, 3)

    # Normalize to [0, 1]
    dino_reduced = (dino_reduced - dino_reduced.min(axis=0)) / (dino_reduced.max(axis=0) - dino_reduced.min(axis=0) + 1e-8)

    return dino_reduced  # (N, 3) in [0, 1]


# Cache the PCA-reduced DINO features globally to avoid recomputing for each frame
_cached_dino_pca = None

def render_dino_features(viewpoint_camera, gaussians, pipe, bg_color):
    """
    Render PCA-reduced DINO features using the standard Gaussian renderer.
    We temporarily replace the RGB colors with PCA-reduced DINO features.
    Returns: (3, H, W) tensor
    """
    global _cached_dino_pca

    if not hasattr(gaussians, '_dino_features') or gaussians._dino_features is None:
        return None

    # Compute PCA once and cache it
    if _cached_dino_pca is None:
        _cached_dino_pca = visualize_dino_features_simple(gaussians)  # (N, 3) numpy

    # Convert to torch
    dino_colors = torch.from_numpy(_cached_dino_pca).float().cuda()  # (N, 3)

    # Temporarily save original colors
    original_features_dc = gaussians._features_dc.clone()
    original_features_rest = gaussians._features_rest.clone()

    # Replace with DINO colors (as DC component only, no SH)
    gaussians._features_dc = torch.nn.Parameter(dino_colors.unsqueeze(1).contiguous())  # (N, 1, 3)
    gaussians._features_rest = torch.nn.Parameter(torch.zeros_like(original_features_rest))  # Zero out SH

    # Render with DINO colors
    try:
        results = render(viewpoint_camera, gaussians, pipe, bg_color)
        dino_rendered = results["render"]  # (3, H, W)
    finally:
        # Restore original colors
        gaussians._features_dc = original_features_dc
        gaussians._features_rest = original_features_rest

    return dino_rendered


def load_and_visualize_sam_mask(feature_field_dir, image_name):
    """
    Load SAM mask from feature-field directory and visualize it.
    Returns: (H, W, 3) RGB visualization of SAM regions
    """
    # Feature-field masks are named like frame_00001.npz
    # But image_name might be like '00000000.png' or 'frame_00001.jpg'
    base_name = os.path.splitext(image_name)[0]

    # Try to find matching file
    possible_names = [
        f"{base_name}.npz",
        f"frame_{base_name}.npz",
        f"frame_{int(base_name):05d}.npz" if base_name.isdigit() else None
    ]

    mask_path = None
    for name in possible_names:
        if name is None:
            continue
        test_path = os.path.join(feature_field_dir, name)
        if os.path.exists(test_path):
            mask_path = test_path
            break

    if mask_path is None:
        return None

    try:
        # Load feature-field mask
        data = np.load(mask_path, allow_pickle=True)

        if 'masks' not in data:
            return None

        # masks: (N_regions, H, W) boolean array
        masks = data['masks']  # (N, H, W)
        N, H, W = masks.shape

        # Create region ID map
        region_ids = np.zeros((H, W), dtype=np.int32)
        for i in range(N):
            region_ids[masks[i]] = i + 1  # Region IDs start from 1

        # Visualize regions with different colors
        rgb_mask = visualize_obj(region_ids.astype(np.uint8))

        return rgb_mask

    except Exception as e:
        print(f"Warning: Failed to load SAM mask for {image_name}: {e}")
        return None

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, feature_field_dir=None):
    """
    Modified render function to visualize:
    1. RGB rendering
    2. Ground truth
    3. DINO features (rendered from 3D Gaussians)
    4. SAM masks (from feature-field pipeline)
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    dino_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dino_features")
    sam_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "sam_masks")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(dino_feature_path, exist_ok=True)
    makedirs(sam_mask_path, exist_ok=True)

    # Check if Gaussians have DINO features
    has_dino = hasattr(gaussians, '_dino_features') and gaussians._dino_features is not None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Render RGB
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :].detach().cpu()

        # Save RGB and GT
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # Render and visualize DINO features if available
        if has_dino:
            # Render DINO features (384-dim -> 3-channel RGB via PCA, done in render_dino_features)
            dino_rendered = render_dino_features(view, gaussians, pipeline, background)  # (3, H, W)
            if dino_rendered is not None:
                # dino_rendered is already RGB, just save it
                torchvision.utils.save_image(dino_rendered, os.path.join(dino_feature_path, '{0:05d}'.format(idx) + ".png"))

        # Load and visualize SAM masks from feature-field if available
        if feature_field_dir is not None:
            sam_viz = load_and_visualize_sam_mask(feature_field_dir, view.image_name)
            if sam_viz is not None:
                Image.fromarray(sam_viz).save(os.path.join(sam_mask_path, '{0:05d}'.format(idx) + ".png"))

    # Create concat video
    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path, exist_ok=True)

    # Determine number of columns based on what's available
    num_cols = 2  # GT + Render
    if has_dino:
        num_cols += 1
    if feature_field_dir is not None and os.path.exists(sam_mask_path) and len(os.listdir(sam_mask_path)) > 0:
        num_cols += 1

    fourcc = cv2.VideoWriter.fourcc(*'DIVX')
    first_gt = np.array(Image.open(os.path.join(gts_path, sorted(os.listdir(gts_path))[0])))
    size = (first_gt.shape[1] * num_cols, first_gt.shape[0])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))

        images_to_concat = [gt, rgb]

        if has_dino:
            dino_path = os.path.join(dino_feature_path, file_name)
            if os.path.exists(dino_path):
                dino_img = np.array(Image.open(dino_path))
                images_to_concat.append(dino_img)

        if feature_field_dir is not None:
            sam_path = os.path.join(sam_mask_path, file_name)
            if os.path.exists(sam_path):
                sam_img = np.array(Image.open(sam_path))
                images_to_concat.append(sam_img)

        result = np.hstack(images_to_concat)
        result = result.astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path, file_name))
        writer.write(result[:,:,::-1])

    writer.release()
    print(f"\nRendering complete. Results saved to {os.path.join(model_path, name, f'ours_{iteration}')}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Get feature-field directory if available
        feature_field_dir = getattr(dataset, 'feature_field_dir', None)
        if feature_field_dir and os.path.exists(feature_field_dir):
            print(f"Using feature-field masks from: {feature_field_dir}")
        else:
            print("No feature-field directory found, SAM masks will not be rendered")
            feature_field_dir = None

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipeline, background, feature_field_dir)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, pipeline, background, feature_field_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

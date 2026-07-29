# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import sys
from pathlib import Path
import torch
from random import randint

from utils.loss_utils import (
    l1_loss,
    ssim,
    loss_cls_3d,
    semantic_feature_supervision_loss,
    knn_semantic_regularization_loss,
)
from utils.enhanced_losses import (
    plane_normal_consistency_loss,
    laplacian_smoothness_loss,
    opacity_binarization_loss,
)

TRAIN_METRICS_FILE = "training_metrics.jsonl"

from gaussian_renderer import render
# network_gui disabled for parallel training
# from gaussian_renderer import network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json

from utils.dino_utils import DINOv2FeatureExtractor
from utils.feature_lifting import (
    lift_dino_features_to_gaussians,
    lift_feature_field_masks_to_gaussians,
)


def _read_config_file(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() in {".yaml", ".yml"}:
                import yaml
                config = yaml.safe_load(file)
            else:
                config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: Failed to parse the JSON configuration file: {exc}")
        exit(1)

    if not isinstance(config, dict):
        print(f"Error: Config file '{path}' must contain a mapping/object.")
        exit(1)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


_GAUSSIAN_KEY_MAP = {
    "densify_from_iter": "densify_from_iter",
    "densify_until_iter": "densify_until_iter",
    "densification_interval": "densification_interval",
    "densify_grad_threshold": "densify_grad_threshold",
    "opacity_reset_interval": "opacity_reset_interval",
    "percent_dense": "percent_dense",
    "position_lr_init": "position_lr_init",
    "position_lr_final": "position_lr_final",
    "position_lr_delay_mult": "position_lr_delay_mult",
    "position_lr_max_steps": "position_lr_max_steps",
    "feature_lr": "feature_lr",
    "opacity_lr": "opacity_lr",
    "scaling_lr": "scaling_lr",
    "rotation_lr": "rotation_lr",
    "lambda_dssim": "lambda_dssim",
    "lambda_semantic": "lambda_semantic",
    "lambda_feature_reg": "lambda_feature_reg",
    "lambda_normal_reg": "lambda_normal_reg",
    "lambda_smoothness_reg": "lambda_smoothness_reg",
    "lambda_opacity_prior": "lambda_opacity_prior",
    "reg3d_interval": "reg3d_interval",
    "reg3d_k": "reg3d_k",
    "reg3d_lambda": "reg3d_lambda_val",
    "reg3d_max_points": "reg3d_max_points",
    "reg3d_sample_size": "reg3d_sample_size",
    "normal_reg_interval": "normal_reg_interval",
    "normal_reg_knn": "normal_reg_knn",
    "smoothness_reg_interval": "smoothness_reg_interval",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "enable_mixed_precision": "enable_mixed_precision",
    "checkpoint_interval_auto_save": "checkpoint_interval_auto_save",
    "num_classes": "num_classes",
    "feature_field_variance": "feature_field_variance",
    "feature_field_variance_alpha": "feature_field_variance_alpha",
    "feature_field_min_views": "feature_field_min_views",
    "geometry_sample_points": "feature_field_matching_geometry_sample_points",
    "geometry_max_points_per_mask": "feature_field_matching_geometry_max_points_per_mask",
    "geometry_weight": "feature_field_matching_geometry_weight",
    "geometry_threshold": "feature_field_matching_geometry_threshold",
    "geometry_semantic_floor": "feature_field_matching_geometry_semantic_floor",
}

_CROSS_VIEW_KEY_MAP = {
    "threshold": "feature_field_matching_threshold",
    "max_view_gap": "feature_field_matching_max_view_gap",
    "topk_per_mask": "feature_field_matching_topk_per_mask",
    "same_view_matching_threshold": "feature_field_same_view_matching_threshold",
    "use_improved_matching": "use_improved_matching",
    "use_multiview_refinement": "use_multiview_refinement",
    "min_consensus_views": "min_consensus_views",
    "filter_mask_ids_by_reliability": "filter_mask_ids_by_reliability",
    "one_to_one": "feature_field_matching_one_to_one",
    "use_geometry_matching": "feature_field_use_geometry_matching",
    "prevent_view_conflicts": "feature_field_prevent_view_conflicts",
}


def load_training_config(config_path: str) -> dict:
    config = _read_config_file(Path(config_path))

    base_path = Path(__file__).resolve().parent / "config" / "pipeline" / "base.yaml"
    if base_path.exists() and Path(config_path).resolve() != base_path.resolve():
        config = _deep_merge(_read_config_file(base_path), config)

    if "gaussian" not in config:
        return config

    gaussian = config.get("gaussian") or {}
    cross_view = config.get("cross_view") or {}
    flattened = {}

    for source_key, target_key in _GAUSSIAN_KEY_MAP.items():
        if source_key in gaussian:
            flattened[target_key] = gaussian[source_key]

    for source_key, target_key in _CROSS_VIEW_KEY_MAP.items():
        if source_key in cross_view:
            flattened[target_key] = cross_view[source_key]

    flattened.setdefault("feature_field_same_view_matching_threshold", -1.0)
    flattened.setdefault("use_improved_matching", True)
    flattened.setdefault("use_multiview_refinement", False)
    flattened.setdefault("min_consensus_views", 2)
    flattened.setdefault("filter_mask_ids_by_reliability", False)
    flattened.setdefault("feature_field_matching_one_to_one", True)
    flattened.setdefault("feature_field_use_geometry_matching", True)
    flattened.setdefault("feature_field_prevent_view_conflicts", True)
    return flattened
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentation import load_feature_field_directory

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    train_cameras = scene.getTrainCameras()

    feature_field_views = None
    feature_field_dir = getattr(dataset, "feature_field_dir", "")
    if feature_field_dir:
        resolved_dir = Path(feature_field_dir).expanduser()
        try:
            resolved_dir = resolved_dir.resolve()
        except FileNotFoundError:
            resolved_dir = resolved_dir

        if resolved_dir.is_dir():
            feature_field_views = load_feature_field_directory(resolved_dir)
            if not feature_field_views:
                feature_field_views = None
        else:
            print(f"Feature-field directory '{resolved_dir}' not found. Falling back to DINO lifting.")

    skip_dino = os.getenv('SKIP_DINO_LIFTING', '0') == '1'
    use_feature_field_masks = feature_field_views is not None

    if not checkpoint:
        if use_feature_field_masks:
            ff_variance = float(getattr(dataset, "feature_field_variance", 0.05))
            ff_min_views = int(getattr(dataset, "feature_field_min_views", 2))
            ff_variance_alpha = float(getattr(dataset, "feature_field_variance_alpha", 2.0))
            ff_matching_threshold = float(getattr(dataset, "feature_field_matching_threshold", 0.7))
            ff_same_view_matching_threshold = float(getattr(dataset, "feature_field_same_view_matching_threshold", -1.0))
            same_view_matching_threshold = ff_same_view_matching_threshold if ff_same_view_matching_threshold >= 0 else None
            use_improved_matching = bool(getattr(dataset, "use_improved_matching", True))
            use_multiview_refinement = bool(getattr(dataset, "use_multiview_refinement", False))
            min_consensus_views = int(getattr(dataset, "min_consensus_views", 2))
            filter_mask_ids_by_reliability = bool(getattr(dataset, "filter_mask_ids_by_reliability", False))
            matching_max_view_gap = int(getattr(dataset, "feature_field_matching_max_view_gap", 10))
            matching_topk_per_mask = int(getattr(dataset, "feature_field_matching_topk_per_mask", 3))
            matching_one_to_one = bool(getattr(dataset, "feature_field_matching_one_to_one", True))
            use_geometry_matching = bool(getattr(dataset, "feature_field_use_geometry_matching", True))
            matching_geometry_sample_points = int(getattr(dataset, "feature_field_matching_geometry_sample_points", 120000))
            matching_geometry_max_points_per_mask = int(getattr(dataset, "feature_field_matching_geometry_max_points_per_mask", 2048))
            matching_geometry_weight = float(getattr(dataset, "feature_field_matching_geometry_weight", 0.35))
            matching_geometry_threshold = float(getattr(dataset, "feature_field_matching_geometry_threshold", 0.05))
            matching_geometry_semantic_floor = float(getattr(dataset, "feature_field_matching_geometry_semantic_floor", 0.60))
            prevent_view_conflicts = bool(getattr(dataset, "feature_field_prevent_view_conflicts", True))

            gaussian_features, reliability_weights, lifting_stats, gaussian_mask_ids = lift_feature_field_masks_to_gaussians(
                gaussians.get_xyz,
                train_cameras,
                feature_field_views,
                variance_threshold=ff_variance,
                min_views=ff_min_views,
                variance_alpha=ff_variance_alpha,
                use_improved_matching=use_improved_matching,
                min_consensus_views=min_consensus_views,
                matching_threshold=ff_matching_threshold,
                same_view_matching_threshold=same_view_matching_threshold,
                use_multiview_refinement=use_multiview_refinement,
                filter_mask_ids_by_reliability=filter_mask_ids_by_reliability,
                matching_max_view_gap=matching_max_view_gap,
                matching_topk_per_mask=matching_topk_per_mask,
                matching_one_to_one=matching_one_to_one,
                use_geometry_matching=use_geometry_matching,
                matching_geometry_sample_points=matching_geometry_sample_points,
                matching_geometry_max_points_per_mask=matching_geometry_max_points_per_mask,
                matching_geometry_weight=matching_geometry_weight,
                matching_geometry_threshold=matching_geometry_threshold,
                matching_geometry_semantic_floor=matching_geometry_semantic_floor,
                prevent_view_conflicts=prevent_view_conflicts,
            )

            actual_dim = gaussian_features.shape[1]
            gaussians.initialize_dino_features(gaussian_features, feature_dim=actual_dim, requires_grad=True)
            gaussians.set_dino_feature_targets(gaussian_features, reliability_weights)

            try:
                device = gaussians.get_xyz.device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if isinstance(gaussian_mask_ids, torch.Tensor):
                gaussians.mask_ids = gaussian_mask_ids.to(device)
            else:
                gaussians.mask_ids = torch.tensor(gaussian_mask_ids, dtype=torch.long, device=device)  # Store 3D mask assignments

        elif not skip_dino:
            # Cache path for DINOv2 features
            dino_cache_path = os.path.join(dataset.model_path, 'dino_features_cache.pth')

            # Step 1: Extract DINOv2 features from all training views
            if os.path.exists(dino_cache_path):
                print(f"Loading cached DINOv2 features from {dino_cache_path}")
                dino_features_2d = torch.load(dino_cache_path)
                extractor = None  # Don't need model if loading from cache
            else:
                print("Extracting DINOv2 features")
                extractor = DINOv2FeatureExtractor(model_name='dinov2_vits14', device='cuda')
                dino_features_2d = extractor.precompute_features_for_dataset(
                    train_cameras,
                    save_path=dino_cache_path
                )

            # Step 2: Lift 2D features to 3D Gaussians (CF3-inspired weighted fusion)
            lift_max_views = int(os.getenv("LIFT_MAX_VIEWS", "150"))
            lift_view_stride = max(1, int(os.getenv("LIFT_VIEW_STRIDE", "1")))
            lift_reduce_dim = int(os.getenv("LIFT_REDUCE_DIM", "384"))
            lift_variance = float(os.getenv("LIFT_VARIANCE_THRESHOLD", "0.05"))
            lift_min_views = int(os.getenv("LIFT_MIN_VIEWS", "2"))

            camera_indices = list(range(0, len(train_cameras), lift_view_stride))
            cameras_for_lifting = [train_cameras[i] for i in camera_indices]
            dino_features_for_lifting = {idx: dino_features_2d[i] for idx, i in enumerate(camera_indices)}

            lift_variance_alpha = float(os.getenv("LIFT_VARIANCE_ALPHA", "3.0"))
            gaussian_features, reliability_weights, lifting_stats = lift_dino_features_to_gaussians(
                gaussians.get_xyz,
                cameras_for_lifting,
                dino_features_for_lifting,
                variance_threshold=lift_variance,
                min_views=lift_min_views,
                max_views=lift_max_views,
                reduce_dim=lift_reduce_dim,
                variance_alpha=lift_variance_alpha,
            )

            # Step 3: Initialize Gaussian DINO features
            actual_dim = gaussian_features.shape[1]
            gaussians.initialize_dino_features(gaussian_features, feature_dim=actual_dim, requires_grad=True)

            gaussians.set_dino_feature_targets(gaussian_features, reliability_weights)

            del dino_features_2d
            if extractor is not None:
                del extractor
            torch.cuda.empty_cache()
        else:
            dummy_dim = 3  # Minimal dimension
            gaussians.initialize_dino_features(
                torch.zeros(gaussians.get_xyz.shape[0], dummy_dim, device='cuda'),
                feature_dim=dummy_dim,
                requires_grad=False
            )
            gaussians.set_dino_feature_targets(None)
    elif skip_dino and not use_feature_field_masks:
        dummy_dim = 3
        gaussians.initialize_dino_features(
            torch.zeros(gaussians.get_xyz.shape[0], dummy_dim, device='cuda'),
            feature_dim=dummy_dim,
            requires_grad=False
        )
        gaussians.set_dino_feature_targets(None)

    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # Gradient accumulation setup
    gradient_accumulation_steps = getattr(opt, 'gradient_accumulation_steps', 1)
    _accumulated_loss = 0.0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Gradient accumulation setup
    gradient_accumulation_steps = getattr(opt, 'gradient_accumulation_steps', 1)
    accumulated_loss = 0.0

    # CUDA events for timing
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()     
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        # Object Loss (only if objects exist)
        if viewpoint_cam.objects is not None:
            gt_obj = viewpoint_cam.objects.cuda().long()
            logits = classifier(objects)
            loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
            loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)
        else:
            loss_obj = torch.tensor(0.0, device="cuda")

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss_obj_3d = None
        logits3d = None
        prob_obj3d = None

        need_logits3d = (iteration % opt.reg3d_interval == 0)
        if opt.lambda_normal_reg > 0 and iteration >= opt.densify_until_iter and iteration % opt.normal_reg_interval == 0:
            need_logits3d = True
        if opt.lambda_smoothness_reg > 0 and iteration >= opt.densify_until_iter and iteration % opt.smoothness_reg_interval == 0:
            need_logits3d = True

        if need_logits3d:
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d, dim=0).squeeze().permute(1,0)

        if iteration % opt.reg3d_interval == 0:
            if prob_obj3d is None:
                prob_obj3d = torch.softmax(classifier(gaussians._objects_dc.permute(2,0,1)), dim=0).squeeze().permute(1,0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj + loss_obj_3d
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj

        loss_semantic = None
        loss_feature_reg = None
        loss_normal_reg = None
        loss_smooth_reg = None
        loss_opacity_prior = None

        if (
            hasattr(gaussians, '_dino_features')
            and gaussians._dino_features.numel() > 0
            and hasattr(gaussians, 'dino_target_features')
            and gaussians.dino_target_features.numel() > 0
            and iteration >= opt.densify_until_iter
        ):
            loss_semantic = semantic_feature_supervision_loss(
                gaussians._dino_features,
                gaussians.dino_target_features,
                reliability_weights=gaussians.dino_reliability_weights,
                lambda_val=1.0,
            )
            loss = loss + opt.lambda_semantic * loss_semantic

            if iteration % opt.reg3d_interval == 0:
                loss_feature_reg = knn_semantic_regularization_loss(
                    gaussians._xyz.squeeze().detach(),
                    gaussians._dino_features,
                    k=opt.reg3d_k,
                    lambda_val=1.0,
                    max_points=opt.reg3d_max_points,
                    sample_size=opt.reg3d_sample_size,
                    reliability_weights=gaussians.dino_reliability_weights,
                )
                loss = loss + opt.lambda_feature_reg * loss_feature_reg

            if logits3d is not None:
                predicted_ids = torch.argmax(logits3d.detach(), dim=0).squeeze()
                if opt.lambda_normal_reg > 0 and iteration % opt.normal_reg_interval == 0:
                    loss_normal_reg = plane_normal_consistency_loss(
                        gaussians._xyz.squeeze(),
                        predicted_ids,
                        k=opt.normal_reg_knn,
                        lambda_val=1.0
                    )
                    loss = loss + opt.lambda_normal_reg * loss_normal_reg

                if opt.lambda_smoothness_reg > 0 and iteration % opt.smoothness_reg_interval == 0:
                    loss_smooth_reg = laplacian_smoothness_loss(
                        gaussians._xyz.squeeze(),
                        predicted_ids,
                        k=opt.normal_reg_knn,
                        lambda_val=1.0
                    )
                    loss = loss + opt.lambda_smoothness_reg * loss_smooth_reg

        if opt.lambda_opacity_prior > 0:
            loss_opacity_prior = opacity_binarization_loss(
                gaussians.get_opacity.squeeze(),
                lambda_val=1.0
            )
            loss = loss + opt.lambda_opacity_prior * loss_opacity_prior

        # Gradient accumulation: scale loss
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        # Clear CUDA cache periodically to prevent OOM
        if iteration % 100 == 0:
            torch.cuda.empty_cache()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                loss_obj_3d,
                use_wandb,
                loss_semantic,
                loss_feature_reg,
                loss_normal_reg,
                loss_smooth_reg,
                loss_opacity_prior
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step (with gradient accumulation support)
            if iteration < opt.iterations and iteration % gradient_accumulation_steps == 0:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                cls_optimizer.step()
                cls_optimizer.zero_grad(set_to_none = True)

            # Auto-save checkpoint periodically for OOM recovery
            checkpoint_interval = getattr(opt, 'checkpoint_interval_auto_save', 5000)
            if iteration % checkpoint_interval == 0:
                checkpoint_path = os.path.join(scene.model_path, f"chkpnt_auto_{iteration}.pth")
                torch.save((gaussians.capture(), iteration), checkpoint_path)
                # Keep only last 2 auto-checkpoints to save disk space
                checkpoints = sorted([f for f in os.listdir(scene.model_path) if f.startswith("chkpnt_auto_")])
                if len(checkpoints) > 2:
                    for old_ckpt in checkpoints[:-2]:
                        os.remove(os.path.join(scene.model_path, old_ckpt))

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    metrics_path = os.path.join(args.model_path, TRAIN_METRICS_FILE)
    if os.path.exists(metrics_path):
        os.remove(metrics_path)


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb, loss_semantic=None, loss_feature_reg=None, loss_normal=None, loss_smooth=None, loss_opacity_prior=None):
    if use_wandb:
        log_dict = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "iter_time": elapsed,
            "iter": iteration
        }
        if loss_obj_3d is not None:
            log_dict["train_loss_patches/loss_obj_3d"] = loss_obj_3d.item()
        if loss_semantic is not None:
            log_dict["train_loss_patches/loss_semantic"] = loss_semantic.item()
        if loss_feature_reg is not None:
            log_dict["train_loss_patches/loss_feature_reg"] = loss_feature_reg.item()
        if loss_normal is not None:
            log_dict["train_loss_patches/loss_normal_reg"] = loss_normal.item()
        if loss_smooth is not None:
            log_dict["train_loss_patches/loss_smooth_reg"] = loss_smooth.item()
        if loss_opacity_prior is not None:
            log_dict["train_loss_patches/loss_opacity_prior"] = loss_opacity_prior.item()
        wandb.log(log_dict)

    metrics_entry = {
        "iteration": int(iteration),
        "l1_loss": float(Ll1.item()),
        "total_loss": float(loss.item()),
        "elapsed_ms": float(elapsed),
    }
    metrics_entry["loss_obj_3d"] = float(loss_obj_3d.item()) if loss_obj_3d is not None else 0.0
    metrics_entry["loss_semantic"] = float(loss_semantic.item()) if loss_semantic is not None else 0.0
    metrics_entry["loss_feature_reg"] = float(loss_feature_reg.item()) if loss_feature_reg is not None else 0.0
    metrics_entry["loss_normal_reg"] = float(loss_normal.item()) if loss_normal is not None else 0.0
    metrics_entry["loss_smooth_reg"] = float(loss_smooth.item()) if loss_smooth is not None else 0.0
    metrics_entry["loss_opacity_prior"] = float(loss_opacity_prior.item()) if loss_opacity_prior is not None else 0.0

    metrics_path = os.path.join(scene.model_path, TRAIN_METRICS_FILE)
    with open(metrics_path, "a") as metrics_file:
        metrics_file.write(json.dumps(metrics_entry) + "\n")
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config/pipeline/example.yaml", help="Path to the training or pipeline YAML config")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    config = load_training_config(args.config_file)

    # Every key returned by load_training_config maps 1:1 to an argparse attribute.
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"  [config] warning: '{key}' has no matching argument; ignored")

    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui disabled for parallel training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb)

    # All done
    print("\nTraining complete.")
TRAIN_METRICS_FILE = "training_metrics.jsonl"

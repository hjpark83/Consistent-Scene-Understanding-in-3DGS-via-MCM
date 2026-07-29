# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial import KDTree

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._objects_dc = torch.empty(0)
        self.num_objects = 16
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        ################################################################################
        # NEW: DINOv2 semantic features (Feature-3DGS inspired)
        # 역할: 각 Gaussian에 semantic feature vector 할당
        # 차원: (N, D) where D=384 for DINOv2-small or 768 for DINOv2-base
        ################################################################################
        self._dino_features = torch.empty(0)
        self.dino_feature_dim = 0  # Will be set during initialization
        self.dino_reliability_weights = torch.empty(0)
        self.dino_target_features = torch.empty(0)
        self.mask_ids = torch.empty(0, dtype=torch.long)
        ################################################################################
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._objects_dc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._dino_features,
            self.dino_feature_dim,
            self.dino_reliability_weights, 
            self.dino_target_features,
            self.mask_ids,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) == 18:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._objects_dc,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._dino_features,
                self.dino_feature_dim,
                self.dino_reliability_weights,
                self.dino_target_features,
                self.mask_ids,
            ) = model_args
        elif len(model_args) == 17:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._objects_dc,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._dino_features,
                self.dino_feature_dim,
                self.dino_reliability_weights,
                self.dino_target_features,
            ) = model_args
            self.mask_ids = torch.empty(0, dtype=torch.long, device=self._xyz.device)
        elif len(model_args) == 16:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._objects_dc,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._dino_features,
                self.dino_feature_dim,
                self.dino_reliability_weights,
            ) = model_args
            self.dino_target_features = self._dino_features.detach().clone()
            self.mask_ids = torch.empty(0, dtype=torch.long, device=self._xyz.device)
        else:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._objects_dc,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._dino_features,
                self.dino_feature_dim,
            ) = model_args
            self.dino_reliability_weights = torch.ones(self._xyz.shape[0], device="cuda")
            self.dino_target_features = self._dino_features.detach().clone() if self._dino_features.numel() > 0 else torch.empty(0)
            self.mask_ids = torch.empty(0, dtype=torch.long, device=self._xyz.device)
        if isinstance(self.mask_ids, torch.Tensor) and self.mask_ids.numel() > 0:
            self.mask_ids = self.mask_ids.to(self._xyz.device).long()
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_objects(self):
        return self._objects_dc

    @property
    def get_dino_features(self):
        return self._dino_features

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # random init obj_id now
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        fused_objects = fused_objects[:,:,None]

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))

    def initialize_dino_features(self, dino_features, feature_dim=384, requires_grad=True):
        N = self.get_xyz.shape[0]

        if dino_features is not None:
            # Use provided features
            assert dino_features.shape[0] == N, f"Feature count mismatch: {dino_features.shape[0]} vs {N}"
            self.dino_feature_dim = dino_features.shape[1]
            self._dino_features = nn.Parameter(dino_features.cuda().requires_grad_(requires_grad))
            self.dino_reliability_weights = torch.ones(N, device="cuda")
        else:
            # Random initialization (fallback)
            self.dino_feature_dim = feature_dim
            random_features = torch.randn(N, feature_dim, device="cuda") * 0.01
            self._dino_features = nn.Parameter(random_features.requires_grad_(requires_grad))
            self.dino_reliability_weights = torch.ones(N, device="cuda")
        self.dino_target_features = self._dino_features.detach().clone()

    def set_dino_feature_targets(self, target_features, reliability_weights=None):
        if target_features is None or target_features.numel() == 0:
            self.dino_target_features = torch.empty(0, device=self.get_xyz.device)
        else:
            self.dino_target_features = target_features.detach().to(self.get_xyz.device)

        if reliability_weights is not None:
            self.dino_reliability_weights = reliability_weights.detach().to(self.get_xyz.device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        if hasattr(self, '_dino_features') and self._dino_features.numel() > 0:
            l.append({'params': [self._dino_features], 'lr': training_args.feature_lr / 2.0, "name": "dino_features"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def finetune_setup(self, training_args, mask3d):
        # Define a function that applies the mask to the gradients
        def mask_hook(grad):
            return grad * mask3d
        def mask_hook2(grad):
            return grad * mask3d.squeeze(-1)
        

        # Register the hook to the parameter (only once!)
        hook_xyz = self._xyz.register_hook(mask_hook2)
        hook_dc = self._features_dc.register_hook(mask_hook)
        hook_rest = self._features_rest.register_hook(mask_hook)
        hook_opacity = self._opacity.register_hook(mask_hook2)
        hook_scaling = self._scaling.register_hook(mask_hook2)
        hook_rotation = self._rotation.register_hook(mask_hook2)

        self._objects_dc.requires_grad = False

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def removal_setup(self, training_args, mask3d):

        mask3d = ~mask3d.bool().squeeze()

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        objects_dc_sub = self._objects_dc[mask3d].detach()


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, False))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, False))
        self._features_rest = nn.Parameter(set_requires_grad(features_rest_sub, False))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, False))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, False))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, False))
        self._objects_dc = nn.Parameter(set_requires_grad(objects_dc_sub, False))


    def inpaint_setup(self, training_args, mask3d):

        def initialize_new_features(features, num_new_points, mask_xyz_values, distance_threshold=0.25, max_distance_threshold=1, k=5):
            """Initialize new points for multiple features based on neighbouring points in the remaining area."""
            new_features = {}
            
            if num_new_points == 0:
                for key in features:
                    new_features[key] = torch.empty((0, *features[key].shape[1:]), device=features[key].device)
                return new_features

            # Get remaining points from features
            remaining_xyz_values = features["xyz"]
            remaining_xyz_values_np = remaining_xyz_values.cpu().numpy()
            
            # Build a KD-Tree for fast nearest-neighbor lookup
            kdtree = KDTree(remaining_xyz_values_np)
            
            # Sample random points from mask_xyz_values as query points
            mask_xyz_values_np = mask_xyz_values.cpu().numpy()
            query_points = mask_xyz_values_np

            # Find the k nearest neighbors in the remaining points for each query point
            distances, indices = kdtree.query(query_points, k=k)
            selected_indices = indices

            # Initialize new points for each feature
            for key, feature in features.items():
                # Convert feature to numpy array
                feature_np = feature.cpu().numpy()
                
                # If we have valid neighbors, calculate the mean of neighbor points
                if feature_np.ndim == 2:
                    neighbor_points = feature_np[selected_indices]
                elif feature_np.ndim == 3:
                    neighbor_points = feature_np[selected_indices, :, :]
                else:
                    raise ValueError(f"Unsupported feature dimension: {feature_np.ndim}")
                new_points_np = np.mean(neighbor_points, axis=1)
                
                # Convert back to tensor
                new_features[key] = torch.tensor(new_points_np, device=feature.device, dtype=feature.dtype)
            
            return new_features['xyz'], new_features['features_dc'], new_features['scaling'], new_features['objects_dc'], new_features['features_rest'], new_features['opacity'], new_features['rotation']
        
        mask3d = ~mask3d.bool().squeeze()
        mask_xyz_values = self._xyz[~mask3d]

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        objects_dc_sub = self._objects_dc[mask3d].detach()

        # Add new points with random initialization
        sub_features = {
            'xyz': xyz_sub,
            'features_dc': features_dc_sub,
            'scaling': scaling_sub,
            'objects_dc': objects_dc_sub,
            'features_rest': features_rest_sub,
            'opacity': opacity_sub,
            'rotation': rotation_sub,
        }

        num_new_points = len(mask_xyz_values)
        with torch.no_grad():
            new_xyz, new_features_dc, new_scaling, new_objects_dc, new_features_rest, new_opacity, new_rotation = initialize_new_features(sub_features, num_new_points, mask_xyz_values)


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(torch.cat([set_requires_grad(xyz_sub, False), set_requires_grad(new_xyz, True)]))
        self._features_dc = nn.Parameter(torch.cat([set_requires_grad(features_dc_sub, False), set_requires_grad(new_features_dc, True)]))
        self._features_rest = nn.Parameter(torch.cat([set_requires_grad(features_rest_sub, False), set_requires_grad(new_features_rest, True)]))
        self._opacity = nn.Parameter(torch.cat([set_requires_grad(opacity_sub, False), set_requires_grad(new_opacity, True)]))
        self._scaling = nn.Parameter(torch.cat([set_requires_grad(scaling_sub, False), set_requires_grad(new_scaling, True)]))
        self._rotation = nn.Parameter(torch.cat([set_requires_grad(rotation_sub, False), set_requires_grad(new_rotation, True)]))
        self._objects_dc = nn.Parameter(torch.cat([set_requires_grad(objects_dc_sub, False), set_requires_grad(new_objects_dc, True)]))

        # for optimize
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Setup optimizer. Only the new points will have gradients.
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"}  # Assuming there's a learning rate for objects_dc in training_args
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc.shape[1]*self._objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        if hasattr(self, '_dino_features') and self._dino_features.numel() > 0:
            for i in range(self._dino_features.shape[1]):
                l.append('dino_{}'.format(i))
        if hasattr(self, 'mask_ids') and self.mask_ids.numel() > 0:
            l.append('mask_id')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        parts = [xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc]

        if hasattr(self, '_dino_features') and self._dino_features.numel() > 0:
            dino_features = self._dino_features.detach().cpu().numpy()
            parts.append(dino_features)

        if hasattr(self, 'mask_ids') and self.mask_ids.numel() > 0:
            mask_ids = self.mask_ids.detach().cpu().numpy().reshape(-1, 1)  # (N, 1)
            parts.append(mask_ids)

        attributes = np.concatenate(parts, axis=1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        dino_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("dino_")]
        if len(dino_names) > 0:
            dino_names = sorted(dino_names, key = lambda x: int(x.split('_')[-1]))
            dino_features = np.zeros((xyz.shape[0], len(dino_names)))
            for idx, attr_name in enumerate(dino_names):
                dino_features[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._dino_features = nn.Parameter(torch.tensor(dino_features, dtype=torch.float, device="cuda").requires_grad_(True))
            self.dino_feature_dim = len(dino_names)
            self.dino_target_features = self._dino_features.detach().clone()
        else:
            self._dino_features = torch.empty(0)
            self.dino_feature_dim = 0
            self.dino_target_features = torch.empty(0)

        # Load mask_ids if present
        if 'mask_id' in [p.name for p in plydata.elements[0].properties]:
            mask_ids = np.asarray(plydata.elements[0]["mask_id"]).astype(np.int64)
            self.mask_ids = torch.tensor(mask_ids, dtype=torch.long, device="cuda")
        else:
            self.mask_ids = torch.empty(0, dtype=torch.long, device="cuda")

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]
        if "dino_features" in optimizable_tensors:
            self._dino_features = optimizable_tensors["dino_features"]
        if self.dino_reliability_weights.numel() > 0:
            self.dino_reliability_weights = self.dino_reliability_weights[valid_points_mask]
        if self.dino_target_features.numel() > 0:
            self.dino_target_features = self.dino_target_features[valid_points_mask]
        if hasattr(self, 'mask_ids') and self.mask_ids.numel() > 0:
            mask_ids_device = self.mask_ids.to(valid_points_mask.device)
            self.mask_ids = mask_ids_device[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc, new_dino_features=None, new_dino_weights=None, new_mask_ids=None, new_dino_targets=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "obj_dc": new_objects_dc}
        
        if new_dino_features is not None:
            d["dino_features"] = new_dino_features

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        if "dino_features" in optimizable_tensors:
            self._dino_features = optimizable_tensors["dino_features"]

        if self.dino_reliability_weights.numel() > 0:
            if new_dino_weights is None:
                new_dino_weights = torch.ones(new_xyz.shape[0], device="cuda")
            else:
                new_dino_weights = new_dino_weights.to(self.dino_reliability_weights.device)
            self.dino_reliability_weights = torch.cat((self.dino_reliability_weights, new_dino_weights), dim=0)

        if self.dino_target_features.numel() > 0:
            if new_dino_targets is None:
                new_dino_targets = torch.zeros((new_xyz.shape[0], self.dino_target_features.shape[1]), device=self.dino_target_features.device)
            else:
                new_dino_targets = new_dino_targets.to(self.dino_target_features.device)
            self.dino_target_features = torch.cat((self.dino_target_features, new_dino_targets), dim=0)

        if hasattr(self, 'mask_ids') and self.mask_ids.numel() > 0:
            if new_mask_ids is not None:
                self.mask_ids = torch.cat((self.mask_ids, new_mask_ids), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1)
        new_dino_features = None
        if hasattr(self, '_dino_features') and self._dino_features.numel() > 0:
            new_dino_features = self._dino_features[selected_pts_mask].repeat(N, 1)
        new_dino_targets = None
        if self.dino_target_features.numel() > 0:
            new_dino_targets = self.dino_target_features[selected_pts_mask].repeat(N, 1)

        new_mask_ids = None
        if hasattr(self, 'mask_ids') and self.mask_ids.numel() > 0:
            mask_ids_device = self.mask_ids.to(selected_pts_mask.device)
            new_mask_ids = mask_ids_device[selected_pts_mask].repeat(N)

        new_dino_weights = None
        if self.dino_reliability_weights.numel() > 0:
            parent_weights = self.dino_reliability_weights[selected_pts_mask]
            new_dino_weights = parent_weights.repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc, new_dino_features, new_dino_weights, new_mask_ids, new_dino_targets)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask]
        new_dino_features = None
        if hasattr(self, '_dino_features') and self._dino_features.numel() > 0:
            new_dino_features = self._dino_features[selected_pts_mask]
        new_dino_targets = None
        if self.dino_target_features.numel() > 0:
            new_dino_targets = self.dino_target_features[selected_pts_mask]

        new_mask_ids = None
        if hasattr(self, 'mask_ids') and self.mask_ids.numel() > 0:
            mask_ids_device = self.mask_ids.to(selected_pts_mask.device)
            new_mask_ids = mask_ids_device[selected_pts_mask]

        new_dino_weights = None
        if self.dino_reliability_weights.numel() > 0:
            new_dino_weights = self.dino_reliability_weights[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc, new_dino_features, new_dino_weights, new_mask_ids, new_dino_targets)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

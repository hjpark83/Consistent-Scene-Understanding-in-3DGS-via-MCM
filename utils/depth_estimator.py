import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2


class DINOv2DepthEstimator:
    def __init__(self, method='depth_anything_v2', device='cuda'):
        self.device = device
        self.method = method
        self._init_depth_anything_v2()

    def _init_depth_anything_v2(self):
        """Initialize Depth Anything V2"""
        try:
            from transformers import pipeline

            # Use Depth Anything V2 Small for speed (can use Base or Large for accuracy)
            self.depth_pipeline = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == 'cuda' else -1
            )

        except Exception as e:
            print(f"Warning: Failed to load Depth Anything V2: {e}")

    @torch.no_grad()
    def estimate(self, image, dino_features=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        H, W = image.size[1], image.size[0]

        return self._estimate_depth_anything_v2(image)

    def _estimate_depth_anything_v2(self, image):
        result = self.depth_pipeline(image)
        depth_tensor = result['predicted_depth']
        depth = depth_tensor.squeeze().cpu().numpy()
        if depth.shape != (image.size[1], image.size[0]):
            depth = cv2.resize(depth, (image.size[0], image.size[1]), interpolation=cv2.INTER_LINEAR)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def compute_depth_gradients(self, depth_map, sigma=1.0):
        depth_smooth = cv2.GaussianBlur(depth_map, (0, 0), sigma)

        grad_x = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=3)

        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        return grad_mag

    def find_depth_boundaries(self, depth_map, threshold=0.1):
        grad_mag = self.compute_depth_gradients(depth_map)
        boundaries = grad_mag > threshold

        return boundaries


if __name__ == "__main__":
    estimator = DINOv2DepthEstimator(method='depth_anything_v2', device='cuda')
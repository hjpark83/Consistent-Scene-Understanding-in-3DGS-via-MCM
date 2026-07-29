from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 may not be available
    cv2 = None


@dataclass
class CannyEdgeDetector:
    """
    Canny edge detection with double threshold and hysteresis.
    """
    low_threshold: float = 50.0
    high_threshold: float = 150.0
    blur_sigma: float = 1.4
    blur_ksize: int = 5
    normalize: bool = True

    def compute(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_grayscale(image).astype(np.uint8)

        if cv2 is not None:
            # Gaussian blur for noise reduction
            blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), self.blur_sigma)

            # Canny edge detection with double threshold
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        else:  # fallback to scipy (less accurate)
            from scipy.ndimage import gaussian_filter, sobel

            blurred = gaussian_filter(gray.astype(np.float32), sigma=self.blur_sigma)

            # Sobel gradients
            gx = sobel(blurred, axis=1)
            gy = sobel(blurred, axis=0)
            magnitude = np.hypot(gx, gy)

            # Simple thresholding (not true Canny)
            edges = ((magnitude > self.low_threshold) * 255).astype(np.uint8)

        # Convert to float [0, 1]
        edge_map = edges.astype(np.float32)

        if not self.normalize:
            return edge_map

        # Normalize to [0, 1]
        max_val = edge_map.max()
        if max_val < 1e-6:
            return np.zeros_like(edge_map)
        return edge_map / max_val

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[..., 0]
        if cv2 is not None:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Manual RGB -> Gray conversion
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return (image[..., :3] * weights).sum(axis=-1)


@dataclass
class LoGEdgeDetector:
    sigma: float = 2.0
    laplacian_ksize: int = 3
    normalize: bool = True

    def compute(self, image: np.ndarray) -> np.ndarray:
        """
        Returns:
            Edge magnitude map normalized to [0, 1] if normalize=True.
        """
        gray = self._to_grayscale(image).astype(np.float32)

        if cv2 is not None:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
            log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=self.laplacian_ksize)
        else:  # fallback to scipy if OpenCV is not present
            from scipy.ndimage import gaussian_filter, laplace

            blurred = gaussian_filter(gray, sigma=self.sigma)
            log = laplace(blurred)

        magnitude = np.abs(log)
        if not self.normalize:
            return magnitude

        min_val = magnitude.min()
        max_val = magnitude.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(magnitude)
        return (magnitude - min_val) / (max_val - min_val)

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[..., 0]
        if cv2 is not None:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Manual RGB -> Gray conversion
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return (image[..., :3] * weights).sum(axis=-1)

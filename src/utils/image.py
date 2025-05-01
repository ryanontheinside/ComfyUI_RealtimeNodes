"""
Image processing utilities for ComfyUI RealTimeNodes.

Contains functions for converting between different image formats,
particularly between ComfyUI tensor format (BHWC) and OpenCV format.
"""

from typing import List, Tuple

import cv2
import numpy as np
import torch
import math
import torch
import torch.nn.functional as F

def flow_to_rgb(self, flow):
    """
    Convert optical flow to RGB visualization similar to torchvision's flow_to_image
    
    Args:
        flow: optical flow tensor of shape [B, 2, H, W]
    Returns:
        RGB visualization tensor of shape [B, 3, H, W]
    """
    B, _, H, W = flow.shape
    
    # Calculate flow magnitude and angle
    flow_x = flow[:, 0]
    flow_y = flow[:, 1]
    magnitude = torch.sqrt(flow_x**2 + flow_y**2)
    angle = torch.atan2(flow_y, flow_x)
    
    # Normalize magnitude for better visualization
    max_mag = torch.max(magnitude.view(B, -1), dim=1)[0].view(B, 1, 1)
    max_mag = torch.clamp(max_mag, min=1e-4)
    magnitude = torch.clamp(magnitude / max_mag, 0, 1)
    
    # Convert angle and magnitude to RGB using HSV->RGB conversion
    # Hue = angle, Saturation = 1, Value = magnitude
    angle_normalized = (angle / (2 * math.pi) + 0.5) % 1.0
    
    # HSV to RGB conversion
    h = angle_normalized * 6
    i = torch.floor(h)
    f = h - i
    p = torch.zeros_like(magnitude)
    q = 1 - f
    t = f
    
    # Initialize RGB channels
    r = torch.zeros_like(magnitude)
    g = torch.zeros_like(magnitude)
    b = torch.zeros_like(magnitude)
    
    # Case 0: h in [0,1)
    mask = (i == 0)
    r[mask] = magnitude[mask]
    g[mask] = magnitude[mask] * t[mask]
    b[mask] = p[mask]
    
    # Case 1: h in [1,2)
    mask = (i == 1)
    r[mask] = magnitude[mask] * q[mask]
    g[mask] = magnitude[mask]
    b[mask] = p[mask]
    
    # Case 2: h in [2,3)
    mask = (i == 2)
    r[mask] = p[mask]
    g[mask] = magnitude[mask]
    b[mask] = magnitude[mask] * t[mask]
    
    # Case 3: h in [3,4)
    mask = (i == 3)
    r[mask] = p[mask]
    g[mask] = magnitude[mask] * q[mask]
    b[mask] = magnitude[mask]
    
    # Case 4: h in [4,5)
    mask = (i == 4)
    r[mask] = magnitude[mask] * t[mask]
    g[mask] = p[mask]
    b[mask] = magnitude[mask]
    
    # Case 5: h in [5,6)
    mask = (i == 5)
    r[mask] = magnitude[mask]
    g[mask] = p[mask]
    b[mask] = magnitude[mask] * q[mask]
    
    # Stack RGB channels
    rgb = torch.stack([r, g, b], dim=1)
    
    return rgb

def gaussian_blur_2d(x, kernel_size=9, sigma=1.0):
    """
    Apply 2D Gaussian blur to input tensor.
    
    Args:
        x (torch.Tensor): Input tensor in BCHW format
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian kernel
    """
    # Create 1D Gaussian kernel
    kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D Gaussian kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    
    # Reshape kernel for convolution
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(x.shape[1], 1, 1, 1)
    
    # Move kernel to same device as input
    kernel_2d = kernel_2d.to(x.device)
    
    # Apply padding
    padding = kernel_size // 2
    
    # Apply convolution
    return F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])

def convert_to_cv2(tensor: torch.Tensor) -> list:
    """Converts a ComfyUI IMAGE tensor (BHWC, float32, 0-1) to a list of cv2 images (BGR, uint8)."""
    if tensor.ndim != 4 or tensor.shape[3] != 3:
        raise ValueError(f"Input tensor must be BHWC with 3 channels, got shape: {tensor.shape}")

    # Convert entire batch to numpy, scale to 0-255, and convert to uint8
    images_np = tensor.cpu().numpy()
    images_np = (images_np * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for cv2
    cv2_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
    return cv2_images


def convert_to_tensor(cv2_images) -> torch.Tensor:
    """Converts cv2 image(s) (BGR, uint8) to a ComfyUI IMAGE tensor (BHWC, float32, 0-1).

    Args:
        cv2_images: A single cv2 image or a list of cv2 images
    """
    # Handle both single image and list of images
    if not isinstance(cv2_images, list):
        cv2_images = [cv2_images]

    # Process each image
    np_images = []
    for cv2_image in cv2_images:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Scale to 0-1
        image_np = rgb_image.astype(np.float32) / 255.0
        np_images.append(image_np)

    # Stack all images into a batch
    batch_np = np.stack(np_images, axis=0)

    # Convert to tensor (already has batch dimension)
    tensor = torch.from_numpy(batch_np)

    return tensor


def create_mask_from_points(height: int, width: int, points: List[Tuple[int, int]], device="cpu") -> torch.Tensor:
    """Creates a binary mask tensor from a list of points using convex hull."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return torch.from_numpy(mask).float().unsqueeze(0).to(device)

    try:
        np_points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(np_points)
        cv2.fillPoly(mask, [hull], 1)
    except Exception:
        return torch.from_numpy(mask).float().unsqueeze(0).to(device)

    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(device)
    return mask_tensor

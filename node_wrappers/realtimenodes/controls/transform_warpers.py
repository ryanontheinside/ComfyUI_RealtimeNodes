"""
Transform warping nodes for applying transformation matrices to images and latents.

These nodes apply affine transformations to various media types in real-time.
"""

import torch
import torch.nn.functional as F

from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.utils.warp import warp_image, warp_latent

#TODO add depth aware throughout

class MatrixTransformImageNode(ControlNodeBase):
    """
    Apply affine transformation matrix to images.
    
    Warps images using a provided transformation matrix.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "image": ("IMAGE",),
            "transform_matrix": ("TRANSFORM_MATRIX",),
            "mode": (
                ["bilinear", "nearest", "bicubic"], 
                {"default": "bilinear", "tooltip": "Interpolation mode for sampling"}
            ),
            "padding_mode": (
                ["zeros", "border", "reflection"], 
                {"default": "border", "tooltip": "How to handle pixels outside the image boundaries"}
            ),
            "align_corners": (
                "BOOLEAN", 
                {"default": True, "tooltip": "Whether to align corners when sampling"}
            ),
        })
        return inputs
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("transformed_image",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/transform"
    
    def update(
        self, 
        image, 
        transform_matrix, 
        mode="bilinear", 
        padding_mode="border", 
        align_corners=True,
        always_execute=True, 
        unique_id=None
    ):
        """
        Apply transformation matrix to an image.
        
        Args:
            image: Input image tensor [B, H, W, C]
            transform_matrix: 3x3 transformation matrix
            mode: Interpolation mode
            padding_mode: Padding mode
            align_corners: Whether to align corners in sampling
            always_execute: Whether to always execute (from ControlNodeBase)
            unique_id: Unique ID for state management (from ControlNodeBase)
            
        Returns:
            Warped image
        """
        # Apply warping to image using utility function
        warped_image = warp_image(
            image, 
            transform_matrix, 
            mode=mode, 
            padding_mode=padding_mode, 
            align_corners=align_corners
        )
        
        return (warped_image,)


class MatrixTransformLatentNode(ControlNodeBase):
    """
    Apply affine transformation matrix to latent space.
    
    Warps latent tensors using a provided transformation matrix.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "latent": ("LATENT",),
            "transform_matrix": ("TRANSFORM_MATRIX",),
            "mode": (
                ["bilinear", "nearest", "bicubic"], 
                {"default": "bilinear", "tooltip": "Interpolation mode for sampling"}
            ),
            "padding_mode": (
                ["zeros", "border", "reflection"], 
                {"default": "border", "tooltip": "How to handle values outside the latent boundaries"}
            ),
            "align_corners": (
                "BOOLEAN", 
                {"default": True, "tooltip": "Whether to align corners when sampling"}
            ),
        })
        return inputs
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("transformed_latent",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/transform"
    
    def update(
        self, 
        latent, 
        transform_matrix, 
        mode="bilinear", 
        padding_mode="border", 
        align_corners=True,
        always_execute=True, 
        unique_id=None
    ):
        """
        Apply transformation matrix to latent space.
        
        Args:
            latent: Input latent dict with 'samples' tensor [B, C, H, W]
            transform_matrix: 3x3 transformation matrix
            mode: Interpolation mode
            padding_mode: Padding mode
            align_corners: Whether to align corners in sampling
            always_execute: Whether to always execute (from ControlNodeBase)
            unique_id: Unique ID for state management (from ControlNodeBase)
            
        Returns:
            Warped latent
        """
        # Apply warping to latent using utility function
        warped_latent = warp_latent(
            latent, 
            transform_matrix, 
            mode=mode, 
            padding_mode=padding_mode, 
            align_corners=align_corners
        )
        
        return (warped_latent,) 
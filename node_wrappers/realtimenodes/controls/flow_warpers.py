"""
Optical flow nodes for generating and applying flow-based warping.

These nodes work with optical flow to create more advanced warping effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.utils.general import AlwaysEqualProxy
from ....src.utils.image import gaussian_blur_2d, flow_to_rgb
from ....src.utils.realtime_flownets import RealTimeFlowNet


class OpticalFlowNode(ControlNodeBase):
    """
    Generate optical flow between images.
    
    Computes the motion flow field between two images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "previous_image": ("IMAGE", {"tooltip": "Previous frame image"}),
            "current_image": ("IMAGE", {"tooltip": "Current frame image"}),
            "flow_scale": (
                "FLOAT", 
                {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, 
                 "tooltip": "Scale factor to apply to flow"}
            ),
            "smoothness": (
                "FLOAT", 
                {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                 "tooltip": "Gaussian blur sigma to smooth flow field"}
            ),
            "visualize": (
                "BOOLEAN", 
                {"default": False, "tooltip": "Output flow visualization as RGB image"}
            ),
        })
        return inputs
        
    RETURN_TYPES = ("FLOW_FIELD", "IMAGE")
    RETURN_NAMES = ("flow", "flow_visualization")
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/flow"
    
    def __init__(self):
        super().__init__()
        self.flow_net = None
    
    def update(
        self, 
        previous_image, 
        current_image, 
        flow_scale=1.0, 
        smoothness=1.0, 
        visualize=False,
        always_execute=True, 
        unique_id=None
    ):
        """
        Generate optical flow between two images.
        
        Args:
            previous_image: Previous frame tensor [B, H, W, C]
            current_image: Current frame tensor [B, H, W, C]
            flow_scale: Factor to scale the flow magnitude
            smoothness: Gaussian blur sigma to apply to the flow field
            visualize: Whether to output a visualization of the flow field
            always_execute: Whether to always execute (from ControlNodeBase)
            unique_id: Unique ID for state management (from ControlNodeBase)
            
        Returns:
            Flow field tensor [B, 2, H, W] and optional visualization
        """
        # Initialize flow network on first use
        if self.flow_net is None:
            self.flow_net = RealTimeFlowNet()
            self.flow_net.eval()
        
        # Ensure inputs are on the same device
        device = previous_image.device
        self.flow_net = self.flow_net.to(device)
        
        # Convert to grayscale and ensure proper format for flow estimation
        # Input is BHWC, convert to BCHW for CNN
        prev_gray = torch.mean(previous_image, dim=-1, keepdim=True).permute(0, 3, 1, 2)
        curr_gray = torch.mean(current_image, dim=-1, keepdim=True).permute(0, 3, 1, 2)
        
        # Stack frames for flow estimation
        stacked = torch.cat([prev_gray, curr_gray], dim=1)
        
        # Estimate flow
        with torch.no_grad():
            flow = self.flow_net(stacked)
        
        # Scale the flow
        flow = flow * flow_scale
        
        # Apply Gaussian blur if smoothness > 0
        if smoothness > 0:
            flow = gaussian_blur_2d(flow, kernel_size=9, sigma=smoothness)
        
        # Create flow visualization if requested
        if visualize:
            flow_rgb = flow_to_rgb(flow)
            # Convert back to BHWC for ComfyUI
            flow_vis = flow_rgb.permute(0, 2, 3, 1)
        else:
            # Create empty tensor for visualization
            flow_vis = torch.zeros_like(previous_image)
        
        # Print flow statistics for debugging
        with torch.no_grad():
            flow_abs_mean = torch.abs(flow).mean().item()
            flow_min = flow.min().item()
            flow_max = flow.max().item()
            print(f"Flow stats - mean abs: {flow_abs_mean:.4f}, min: {flow_min:.4f}, max: {flow_max:.4f}")
        
        return (flow, flow_vis)


class FlowWarpNode(ControlNodeBase):
    """
    Apply optical flow field to warp images or latents.
    
    Uses flow fields for advanced warping effects.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "data_type": (
                ["IMAGE", "LATENT"], 
                {"default": "IMAGE", "tooltip": "Type of data to warp"}
            ),
            "data": (
                AlwaysEqualProxy("*"), 
                {"tooltip": "Image or latent to warp with flow field"}
            ),
            "flow_field": (
                "FLOW_FIELD", 
                {"tooltip": "Flow field tensor [B, 2, H, W]"}
            ),
            "strength": (
                "FLOAT", 
                {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1,
                 "tooltip": "Strength of the flow warping effect (try higher values if no effect is visible)"}
            ),
            "mode": (
                ["bilinear", "nearest", "bicubic"], 
                {"default": "bilinear", "tooltip": "Interpolation mode for sampling"}
            ),
            "padding_mode": (
                ["zeros", "border", "reflection"], 
                {"default": "border", "tooltip": "How to handle pixels outside boundaries"}
            ),
            "align_corners": (
                "BOOLEAN", 
                {"default": True, "tooltip": "Whether to align corners in sampling grid"}
            ),
            "debug_mode": (
                ["none", "flow", "grid", "test_pattern"], 
                {"default": "none", "tooltip": "Visualization mode for debugging"}
            ),
        })
        return inputs
        
    RETURN_TYPES = (AlwaysEqualProxy("*"), "IMAGE")
    RETURN_NAMES = ("warped_data", "debug_visualization")
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/flow"
    
    def create_test_pattern(self, height, width, device):
        """Create a test pattern grid image to visualize warping effects"""
        # Create checkerboard pattern
        y, x = torch.meshgrid(torch.linspace(0, height-1, height, device=device),
                              torch.linspace(0, width-1, width, device=device),
                              indexing="ij")
        
        # Create grid lines
        grid_size = 32
        grid_x = (x % grid_size < 2).float()
        grid_y = (y % grid_size < 2).float()
        grid = grid_x + grid_y
        
        # Create circles
        centers = [(height//4, width//4), (height//4, 3*width//4), 
                   (3*height//4, width//4), (3*height//4, 3*width//4)]
        circles = torch.zeros_like(grid)
        for cy, cx in centers:
            dist = torch.sqrt((y - cy)**2 + (x - cx)**2)
            circles += (dist < 30).float()
        
        # Combine patterns
        r = grid
        g = (x / width)
        b = (y / height)
        
        # Add circles
        r = torch.clamp(r + circles, 0, 1)
        
        # Stack channels
        pattern = torch.stack([r, g, b], dim=-1)
        
        # Add batch dimension
        pattern = pattern.unsqueeze(0)
        
        return pattern
    
    def update(
        self, 
        data_type, 
        data, 
        flow_field, 
        strength=5.0, 
        mode="bilinear", 
        padding_mode="border",
        align_corners=True,
        debug_mode="none",
        always_execute=True, 
        unique_id=None
    ):
        """
        Warp image or latent using optical flow field.
        
        Args:
            data_type: Type of data to warp ('IMAGE' or 'LATENT')
            data: Image tensor [B, H, W, C] or latent dict
            flow_field: Flow field tensor [B, 2, H, W]
            strength: Strength of the flow warping effect
            mode: Interpolation mode
            padding_mode: Padding mode
            align_corners: Whether to align corners in sampling grid
            debug_mode: Visualization mode for debugging
            always_execute: Whether to always execute (from ControlNodeBase)
            unique_id: Unique ID for state management (from ControlNodeBase)
            
        Returns:
            Warped image or latent, and debug visualization
        """
        # Print debug info
        print(f"FlowWarpNode - data_type: {data_type}, strength: {strength}")
        print(f"Flow field shape: {flow_field.shape}, dtype: {flow_field.dtype}, device: {flow_field.device}")
        if data_type == "IMAGE":
            print(f"Image shape: {data.shape}, dtype: {data.dtype}, device: {data.device}")
        
        # Flow statistics
        with torch.no_grad():
            flow_abs_mean = torch.abs(flow_field).mean().item()
            flow_min = flow_field.min().item()
            flow_max = flow_field.max().item()
            print(f"Flow stats - mean abs: {flow_abs_mean:.4f}, min: {flow_min:.4f}, max: {flow_max:.4f}")
        
        # Scale flow by strength
        scaled_flow = flow_field * strength
        
        # Create debug visualization based on mode
        if data_type == "IMAGE":
            if debug_mode == "flow":
                # Flow RGB visualization
                flow_rgb = flow_to_rgb(scaled_flow)
                debug_vis = flow_rgb.permute(0, 2, 3, 1)  # BCHW -> BHWC
            elif debug_mode == "test_pattern":
                # Use test pattern to clearly see warping effect
                B, H, W, C = data.shape
                test_pattern = self.create_test_pattern(H, W, device=data.device)
                data = test_pattern  # Replace input with test pattern
                # Also show this as debug visualization
                debug_vis = test_pattern
            else:
                # No specific debug visualization
                debug_vis = torch.zeros_like(data)
        else:
            # For latent, just create a placeholder visualization
            debug_vis = torch.zeros((1, 128, 128, 3), device=flow_field.device)
        
        # Process based on data type
        if data_type == "IMAGE":
            B, H, W, C = data.shape
            device = data.device
            
            # Create coordinate grid in pixel space (0 to H-1, 0 to W-1)
            y_coords = torch.arange(0, H, device=device).float()
            x_coords = torch.arange(0, W, device=device).float()
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            pixel_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
            
            # Apply flow to pixel coordinates (forward warping)
            # But we need backward warping for grid_sample, so we need to convert
            # Flow is "where pixels move to" but we need "where to sample from"
            
            # Convert flow from BCHW [B, 2, H, W] to BHWC [B, H, W, 2]
            flow_grid = scaled_flow.permute(0, 2, 3, 1).clone()  # Clone to avoid in-place issues
            
            # Apply flow to destination coordinates
            # dst_coords = pixel_grid + flow_grid
            
            # For grid_sample, we need normalized coords in [-1, 1]
            # First, build a basic normalized grid
            norm_y = 2.0 * torch.arange(H, device=device) / (H - 1) - 1.0
            norm_x = 2.0 * torch.arange(W, device=device) / (W - 1) - 1.0
            norm_grid_y, norm_grid_x = torch.meshgrid(norm_y, norm_x, indexing="ij")
            identity_grid = torch.stack([norm_grid_x, norm_grid_y], dim=-1)  # [H, W, 2]
            
            # Expand to batch dimension
            identity_grid = identity_grid.expand(B, H, W, 2)
            
            # For grid_sample, we need to know where to sample from
            # Flow tells us where pixels move to, so we reverse it
            # Normalize flow to [-1, 1] range
            flow_grid[..., 0] = flow_grid[..., 0] * (2.0 / (W - 1))  # x component
            flow_grid[..., 1] = flow_grid[..., 1] * (2.0 / (H - 1))  # y component
            
            # Build the sampling grid by subtracting normalized flow from identity grid
            sampling_grid = identity_grid - flow_grid
            
            if debug_mode == "grid" and data_type == "IMAGE":
                # Visualize grid distortion
                debug_img = torch.zeros((B, H, W, 3), device=device)
                
                # Normalize grid to [0, 1] for visualization
                grid_vis = (sampling_grid + 1) / 2.0
                
                # Assign x and y components to R and G channels
                debug_img[..., 0] = grid_vis[..., 0]  # R = x
                debug_img[..., 1] = grid_vis[..., 1]  # G = y
                
                # Draw grid lines
                grid_size = 0.1
                grid_x = torch.abs(torch.frac(grid_vis[..., 0] / grid_size) - 0.5) < 0.02
                grid_y = torch.abs(torch.frac(grid_vis[..., 1] / grid_size) - 0.5) < 0.02
                debug_img[..., 2] = torch.logical_or(grid_x, grid_y).float()
                
                debug_vis = debug_img
            
            # Convert image from BHWC to BCHW for grid_sample
            image_bchw = data.permute(0, 3, 1, 2)
            
            # Apply grid sampling
            warped_image_bchw = F.grid_sample(
                image_bchw, 
                sampling_grid, 
                mode=mode, 
                padding_mode=padding_mode, 
                align_corners=align_corners
            )
            
            # Convert back to BHWC
            warped_image = warped_image_bchw.permute(0, 2, 3, 1)
            
            # For test pattern, return both the warped pattern and the original
            if debug_mode == "test_pattern":
                # Show side-by-side comparison
                combined = torch.cat([data, warped_image], dim=2)  # Concatenate along width
                debug_vis = combined
            
            return (warped_image, debug_vis)
            
        elif data_type == "LATENT":
            # Handle LATENT data type
            if not isinstance(data, dict) or "samples" not in data:
                raise ValueError("Expected latent input to be a dictionary with 'samples' key")
                
            samples = data["samples"]
            B, C, H, W = samples.shape
            device = samples.device
            
            # Resize flow to match latent dimensions if needed
            if flow_field.shape[2:] != samples.shape[2:]:
                scaled_flow = F.interpolate(
                    scaled_flow, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Build the sampling grid
            norm_y = 2.0 * torch.arange(H, device=device) / (H - 1) - 1.0
            norm_x = 2.0 * torch.arange(W, device=device) / (W - 1) - 1.0
            norm_grid_y, norm_grid_x = torch.meshgrid(norm_y, norm_x, indexing="ij")
            identity_grid = torch.stack([norm_grid_x, norm_grid_y], dim=-1)  # [H, W, 2]
            
            # Expand to batch dimension
            identity_grid = identity_grid.expand(B, H, W, 2)
            
            # Convert flow from BCHW [B, 2, H, W] to BHWC [B, H, W, 2]
            flow_grid = scaled_flow.permute(0, 2, 3, 1).clone() 
            
            # Normalize flow to [-1, 1] range
            flow_grid[..., 0] = flow_grid[..., 0] * (2.0 / (W - 1))  # x component
            flow_grid[..., 1] = flow_grid[..., 1] * (2.0 / (H - 1))  # y component
            
            # Build the sampling grid by subtracting normalized flow from identity grid
            sampling_grid = identity_grid - flow_grid
            
            # Apply grid sampling (latent is already in BCHW format)
            warped_samples = F.grid_sample(
                samples, 
                sampling_grid, 
                mode=mode, 
                padding_mode=padding_mode, 
                align_corners=align_corners
            )
            
            # Create new latent dict with warped samples
            warped_latent = {k: v for k, v in data.items()}  # Shallow copy
            warped_latent["samples"] = warped_samples
            
            return (warped_latent, debug_vis)
            
        else:
            raise ValueError(f"Unsupported data type: {data_type}") 
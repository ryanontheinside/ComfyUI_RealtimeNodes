import torch
import numpy as np
from .base.control_base import ControlNodeBase

class DepthMapWarpNode(ControlNodeBase):
    """
    A node that warps any depth map with a radial stretch and optional pulsing effect, designed for gamepad control.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "depth_map": ("IMAGE", {
                    "tooltip": "Input depth map to warp (BHWC format, any content)"
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "X coordinate of the warp center (0-1), e.g., left joystick X"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Y coordinate of the warp center (0-1), e.g., left joystick Y"
                }),
                "stretch_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Strength of the stretch (positive = outward, negative = inward), e.g., right trigger - left trigger"
                }),
                "falloff": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Controls how quickly the stretch diminishes, e.g., right joystick Y"
                }),
                "pulse_frequency": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Frequency of pulsing effect (0 = off), e.g., right joystick X"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask to limit where the warp is applied (0-1 values)"
                }),
            "mode": (["radial", "stretch", "bend", "wave"], {"default": "radial", "tooltip": "Warp mode"}),

            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "update"
    CATEGORY = "image/transforms"
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_count = 0  # For pulsing effect

    def update(self, depth_map, center_x, center_y, stretch_strength, falloff, pulse_frequency, mode, mask=None):
        self.frame_count += 1
        depth_map = depth_map.to(self.device)
        batch_size, height, width, channels = depth_map.shape

        # Create coordinate grid
        y = torch.linspace(0, 1, height, device=self.device)
        x = torch.linspace(0, 1, width, device=self.device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        dx = x_grid - center_x
        dy = y_grid - center_y
        distance = torch.sqrt(dx**2 + dy**2)
        max_distance = torch.tensor(1.414, device=self.device)
        distance = distance / max_distance

        # Apply pulsing effect if enabled
        effective_strength = stretch_strength
        if pulse_frequency > 0:
            pulse = torch.sin(torch.tensor(self.frame_count * pulse_frequency * 0.1, device=self.device))
            effective_strength = stretch_strength * (1 + 0.5 * pulse)

        # Compute warp based on mode
        if mode == "radial":  # Original radial stretch
            displacement = effective_strength * torch.exp(-falloff * distance)
            warp_x = x_grid + dx * displacement
            warp_y = y_grid + dy * displacement

        elif mode == "stretch":  # Vertical or horizontal stretch
            if effective_strength > 0:  # Positive = vertical stretch
                displacement = effective_strength * torch.exp(-falloff * torch.abs(dy))  # Stretch along y-axis
                warp_x = x_grid
                warp_y = y_grid + dy * displacement
            else:  # Negative = horizontal stretch
                displacement = -effective_strength * torch.exp(-falloff * torch.abs(dx))  # Stretch along x-axis
                warp_x = x_grid + dx * displacement
                warp_y = y_grid

        elif mode == "bend":  # Bend left or right from vertical axis
            bend_factor = effective_strength * torch.exp(-falloff * torch.abs(dx))  # Bend strength decreases with x-distance
            warp_x = x_grid + bend_factor * (y_grid - center_y)**2  # Quadratic bend along y
            warp_y = y_grid

        elif mode == "wave":  # Horizontal traveling wave
            wave = effective_strength * torch.sin(10 * (x_grid - center_x) + self.frame_count * 0.2) * torch.exp(-falloff * distance)
            warp_x = x_grid + wave
            warp_y = y_grid

        # Clamp coordinates to [0, 1]
        warp_x = torch.clamp(warp_x, 0, 1)
        warp_y = torch.clamp(warp_y, 0, 1)

        # Convert to grid sample coordinates (-1 to 1)
        grid = torch.stack((warp_x * 2 - 1, warp_y * 2 - 1), dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Sample the warped depth map
        warped_depth = torch.nn.functional.grid_sample(
            depth_map.permute(0, 3, 1, 2),  # BCHW
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        warped_depth = warped_depth.permute(0, 2, 3, 1)  # BHWC

        # Apply mask if provided
        if mask is not None:
            mask = mask.to(self.device)
            if mask.dim() == 3:  # BHW
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 2:  # HW
                mask = mask.unsqueeze(0).unsqueeze(-1)
            warped_depth = depth_map * (1 - mask) + warped_depth * mask

        return (warped_depth.clamp(0, 1),)
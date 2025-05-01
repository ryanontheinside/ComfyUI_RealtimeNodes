import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ....src.utils.image import gaussian_blur_2d, flow_to_rgb
from ....src.utils.realtime_flownets import RealTimeFlowNet
#NOTE: this is totally experimental and grounded in very little research.



class TemporalNetV2Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "current_frame": ("IMAGE",),
                "previous_frame": ("IMAGE",),
                "flow_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"
    CATEGORY = "controlnet/preprocessors"

    def __init__(self):
        self.flow_net = RealTimeFlowNet()
        self.flow_net.eval()
        
    def preprocess(self, current_frame, previous_frame, flow_scale=1.0, blur_sigma=1.0):
        # Ensure inputs are on the same device
        device = current_frame.device
        self.flow_net = self.flow_net.to(device)
        
        # Convert to grayscale and ensure proper format
        # Input is BHWC, convert to BCHW for CNN
        current_gray = torch.mean(current_frame, dim=-1, keepdim=True).movedim(-1, 1)
        prev_gray = torch.mean(previous_frame, dim=-1, keepdim=True).movedim(-1, 1)
        
        # Stack frames for flow estimation
        stacked = torch.cat([prev_gray, current_gray], dim=1)
        
        # Estimate flow
        with torch.no_grad():
            flow = self.flow_net(stacked)
            
        # Scale the flow
        flow = flow * flow_scale
        
        # Apply Gaussian blur if needed
        if blur_sigma > 0:
            flow = gaussian_blur_2d(flow, kernel_size=9, sigma=blur_sigma)
        
        # Visualize flow as RGB (similar to flow_to_image in torchvision)
        # This converts 2-channel flow to 3-channel RGB visualization
        flow_rgb = self.flow_to_rgb(flow)
        
        # Convert flow back to BHWC format
        flow_rgb = flow_rgb.movedim(1, -1)
        
        # Convert previous frame to correct format if needed (ensure BHWC)
        if len(previous_frame.shape) == 3:
            previous_frame = previous_frame.unsqueeze(0)
        
        # Combine previous frame RGB and flow RGB into 6-channel tensor (matching reference)
        combined = torch.cat([
            previous_frame,  # Previous frame RGB (3 channels)
            flow_rgb,        # Flow visualization (3 channels)
        ], dim=-1)
        
        return (combined,)
    

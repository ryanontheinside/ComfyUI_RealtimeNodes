import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.utils.general import AlwaysEqualProxy

class StateResetNode(ControlNodeBase):
    """Node that resets all control node states when triggered"""

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({"trigger": ("BOOLEAN", {"default": False, "tooltip": "Set to True to reset all states"})})
        return inputs

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/utility"

    def update(self, trigger, always_execute=True):
        if trigger:
            self.state_manager.clear_all_states()
            return (True,)
        return (False,)


class StateTestNode(ControlNodeBase):
    """Simple node that maintains a counter to test state management"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update(
            {
                "increment": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Amount to increment counter by"},
                )
            }
        )
        return inputs

    RETURN_TYPES = ("INT",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/utility"

    def update(self, increment, always_execute=True):
        state = self.get_state({"counter": 0})

        state["counter"] += increment

        self.set_state(state)

        return (state["counter"],)


class GetStateNode(ControlNodeBase):
    """
    Node that retrieves a value from the global state using a user-specified key.
    """

    CATEGORY = "utils"
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "update"
    DESCRIPTION = "Retrieve a value from the global state using the given key. If the key is not found, return the default value."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update(
            {
                "key": (
                    "STRING",
                    {
                        "default": "default_key",
                        "tooltip": "The key to retrieve the value from. If not provided, the default value will be returned.",
                    },
                ),
                "default_value": (AlwaysEqualProxy("*"), {"tooltip": "The value to return if the key is not found."}),
                "use_default": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True, the default value will be returned if the key is not found.",
                    },
                ),
            }
        )
        return inputs

    def update(self, key: str, default_value, use_default: bool, always_execute=True):
        """
        Retrieve a value from the global state using the given key.
        """
        if not key or use_default:
            return (default_value,)

        # Get the shared state dictionary
        shared_state = self.state_manager.get_state("__shared_keys__", {})

        # Check if the key exists
        if key in shared_state:
            return (shared_state[key],)

        # Return default value if key not found
        return (default_value,)


class SetStateNode(ControlNodeBase):
    """
    Node that stores a value in the global state with a user-specified key.
    The value will be accessible in future workflow runs through GetStateNode.
    """

    CATEGORY = "utils"
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "update"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Store a value in the global state with the given key. The value will be accessible in future workflow runs through GetStateNode."
    )

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update(
            {
                "key": (
                    "STRING",
                    {
                        "default": "default_key",
                        "tooltip": "The key to store the value under. If not provided, the value will not be stored.",
                    },
                ),
                "value": (AlwaysEqualProxy("*"), {"tooltip": "The value to store in the global state."}),
            }
        )
        return inputs

    def update(self, key: str, value, always_execute=True):
        """
        Store a value in the global state with the given key.
        """
        if not key:
            return (value,)

        try:
            shared_state = self.state_manager.get_state("__shared_keys__", {})
            shared_state[key] = copy.deepcopy(value)
            self.state_manager.set_state("__shared_keys__", shared_state)
        except Exception as e:
            print(f"[State Node] Error storing value: {str(e)}")

        return (value,)

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

class RealTimeFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Lightweight CNN for real-time flow estimation
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, 3, padding=1)
        
        # Initialize weights for better flow estimation
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # x should be in BCHW format
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

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



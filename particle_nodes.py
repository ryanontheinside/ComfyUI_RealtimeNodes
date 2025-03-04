import torch
import numpy as np
from .base.control_base import ControlNodeBase

class TemporalParticleDepthNode(ControlNodeBase):
    """
    A node that generates temporally consistent particle depth maps with a controllable origin point.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "origin_x": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "display": "slider",
                "tooltip": "X coordinate of the origin point (0-1)"
            }),
            "origin_y": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "display": "slider",
                "tooltip": "Y coordinate of the origin point (0-1)"
            }),
            "width": ("INT", {
                "default": 512,
                "min": 64,
                "max": 2048,
                "step": 8,
                "tooltip": "Width of the output depth map"
            }),
            "height": ("INT", {
                "default": 512,
                "min": 64,
                "max": 2048,
                "step": 8,
                "tooltip": "Height of the output depth map"
            }),
            "num_particles": ("INT", {
                "default": 100,
                "min": 10,
                "max": 1000,
                "step": 10,
                "tooltip": "Number of particles to simulate"
            }),
            "particle_size": ("FLOAT", {
                "default": 0.03,
                "min": 0.001,
                "max": 0.2,
                "step": 0.001,
                "tooltip": "Size of each particle (relative to image size)"
            }),
        })
        inputs["optional"] = {
            "hand_data": ("HAND_DATA", {
                "tooltip": "Optional hand tracking data to use for origin point"
            }),
            "hand_keypoint": (["palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"], {
                "default": "index_tip",
                "tooltip": "Which hand keypoint to use as origin (if hand_data is provided)"
            }),
        }
        return inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "update"
    CATEGORY = "image/generators"
    
    def __init__(self):
        super().__init__()
        self.generator = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def update(self, origin_x, origin_y, width, height, num_particles, particle_size, always_execute=True, hand_data=None, hand_keypoint="index_tip"):
        # Initialize generator if needed or if parameters changed
        if self.generator is None:
            self.generator = TemporalParticleDepthGenerator(
                batch_size=1,
                height=height,
                width=width,
                num_particles=num_particles,
                initial_origin_x=origin_x,
                initial_origin_y=origin_y,
                particle_size=particle_size,
                device=self.device
            )
        elif (self.generator.shape[1] != height or 
              self.generator.shape[2] != width or 
              self.generator.num_particles != num_particles):
            # Reinitialize if dimensions or particle count changed
            self.generator = TemporalParticleDepthGenerator(
                batch_size=1,
                height=height,
                width=width,
                num_particles=num_particles,
                initial_origin_x=origin_x,
                initial_origin_y=origin_y,
                particle_size=particle_size,
                device=self.device
            )
        
        # Use hand data if provided
        if hand_data is not None and len(hand_data) > 0:
            # Extract the first hand's data
            hand_info = hand_data[0]
            
            # Check if hands are present
            if hand_info["hands_present"]:
                # Determine which hand to use (prefer right hand if available)
                hand_landmarks = None
                if hand_info["right_hand"] is not None:
                    hand_landmarks = hand_info["right_hand"]
                elif hand_info["left_hand"] is not None:
                    hand_landmarks = hand_info["left_hand"]
                
                if hand_landmarks is not None:
                    # Map keypoint to index
                    keypoint_map = {
                        "palm": 0,       # Center of palm
                        "thumb_tip": 4,  # Thumb tip
                        "index_tip": 8,  # Index finger tip
                        "middle_tip": 12, # Middle finger tip
                        "ring_tip": 16,  # Ring finger tip
                        "pinky_tip": 20  # Pinky tip
                    }
                    
                    if hand_keypoint in keypoint_map:
                        keypoint_idx = keypoint_map[hand_keypoint]
                        # Get normalized coordinates (already in 0-1 range)
                        if keypoint_idx < len(hand_landmarks):
                            # X and Y coordinates are at indices 0 and 1
                            origin_x = hand_landmarks[keypoint_idx][0]
                            origin_y = hand_landmarks[keypoint_idx][1]
        
        # Generate depth map with current origin
        depth_map = self.generator.update(
            new_origin_x=origin_x,
            new_origin_y=origin_y,
            particle_size=particle_size
        )
        
        # ComfyUI already uses BHWC format, so no permute needed
        return (depth_map,)


class TemporalParticleDepthGenerator:
    def __init__(self, 
                 batch_size: int, 
                 height: int, 
                 width: int, 
                 num_particles: int = 100,
                 initial_origin_x: float = 0.5, 
                 initial_origin_y: float = 0.5,
                 particle_size: float = 0.03,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize a temporally consistent particle depth map generator.
        """
        self.device = torch.device(device)
        self.shape = (batch_size, height, width)
        self.particle_size = particle_size
        
        # Coordinate grid (static)
        y = torch.linspace(0, 1, height, device=device)
        x = torch.linspace(0, 1, width, device=device)
        self.y_grid, self.x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # Persistent particle properties
        self.num_particles = num_particles
        self.angles = torch.rand(num_particles, device=device) * 2 * torch.pi
        self.speeds = torch.rand(num_particles, device=device) * 0.02 + 0.005  # Vary speed slightly
        self.distances = torch.zeros(num_particles, device=device)  # Start at origin
        
        # Origin can be updated later
        self.origin_x = torch.tensor(initial_origin_x, device=device)
        self.origin_y = torch.tensor(initial_origin_y, device=device)
        
        # For respawning particles that go off-screen
        self.active = torch.ones(num_particles, dtype=torch.bool, device=device)

    def update(self, 
               new_origin_x: float = None, 
               new_origin_y: float = None, 
               particle_size: float = None) -> torch.Tensor:
        """
        Update particle positions and generate next depth map frame.
        
        Args:
            new_origin_x (float, optional): Update origin X (0-1)
            new_origin_y (float, optional): Update origin Y (0-1)
            particle_size (float, optional): Update particle size
        
        Returns:
            torch.Tensor: Depth map (B, H, W, 3)
        """
        # Update origin if provided
        if new_origin_x is not None:
            self.origin_x = torch.tensor(new_origin_x, device=self.device)
        if new_origin_y is not None:
            self.origin_y = torch.tensor(new_origin_y, device=self.device)
        if particle_size is not None:
            self.particle_size = particle_size

        # Update distances
        self.distances = self.distances + self.speeds
        
        # Check which particles need respawning
        off_screen = self.distances > 1.414  # Beyond sqrt(2), the diagonal
        if off_screen.any():
            num_respawn = off_screen.sum()
            self.distances[off_screen] = 0.0
            # Optional: Randomize angles for respawned particles
            self.angles[off_screen] = torch.rand(num_respawn, device=self.device) * 2 * torch.pi
            self.speeds[off_screen] = torch.rand(num_respawn, device=self.device) * 0.02 + 0.005

        # Calculate current particle positions
        particle_x = self.origin_x + torch.cos(self.angles) * self.distances
        particle_y = self.origin_y + torch.sin(self.angles) * self.distances
        
        # Generate depth map
        depth = torch.zeros(self.shape, device=self.device)
        for i in range(self.num_particles):
            # Skip if particle is inactive (future feature)
            if not self.active[i]:
                continue
                
            # Distance from particle center
            dist = torch.sqrt((self.x_grid - particle_x[i])**2 + 
                            (self.y_grid - particle_y[i])**2)
            
            # Gaussian particle shape
            particle = torch.exp(-(dist**2) / (2 * self.particle_size**2))
            
            # Depth based on distance from origin
            depth[0] = torch.maximum(depth[0], particle * self.distances[i])

        # Normalize and expand to BHWC
        depth = depth / (depth.max() + 1e-6)
        return depth.unsqueeze(-1).expand(-1, -1, -1, 3).clamp(0, 1)
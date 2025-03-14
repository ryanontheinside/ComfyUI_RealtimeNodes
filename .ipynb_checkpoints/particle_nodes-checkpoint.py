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
            "speed": ("FLOAT", {
                "default": 1.0,
                "min": 0.1,
                "max": 3.0,
                "step": 0.1,
                "display": "slider",
                "tooltip": "Overall movement speed of particles"
            }),
            "inertia_factor": ("FLOAT", {
                "default": 0.1,
                "min": 0.01,
                "max": 0.5,
                "step": 0.01,
                "display": "slider",
                "tooltip": "Controls how quickly particles respond to changes (lower = more inertia)"
            }),
            "distance_response": ("FLOAT", {
                "default": 0.15,
                "min": -0.3,
                "max": 0.3,
                "step": 0.01,
                "display": "slider",
                "tooltip": "How distance affects response speed (positive = closer particles respond faster, negative = distant particles respond faster, zero = uniform response)"
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
    
    def update(self, origin_x, origin_y, width, height, num_particles, particle_size, speed, inertia_factor, distance_response, always_execute=True, hand_data=None, hand_keypoint="index_tip"):
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
                speed=speed,
                inertia_factor=inertia_factor,
                distance_response=distance_response,
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
                speed=speed,
                inertia_factor=inertia_factor,
                distance_response=distance_response,
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
            particle_size=particle_size,
            speed=speed,
            inertia_factor=inertia_factor,
            distance_response=distance_response
        )
        
        # ComfyUI already uses BHWC format, so no permute needed
        return (depth_map,)


import torch

class TemporalParticleDepthGenerator:
    def __init__(self, 
                 batch_size: int, 
                 height: int, 
                 width: int, 
                 num_particles: int = 100,
                 initial_origin_x: float = 0.5, 
                 initial_origin_y: float = 0.5,
                 particle_size: float = 0.03,
                 speed: float = 1.0,
                 inertia_factor: float = 0.1,
                 distance_response: float = 0.15,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize a temporally consistent particle depth map generator with inertial origin movement.
        """
        self.device = torch.device(device)
        self.shape = (batch_size, height, width)
        self.particle_size = particle_size
        self.speed = speed
        self.base_inertia_factor = inertia_factor
        self.distance_response = distance_response
        
        # Coordinate grid (static)
        y = torch.linspace(0, 1, height, device=device)
        x = torch.linspace(0, 1, width, device=device)
        self.y_grid, self.x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # Persistent particle properties
        self.num_particles = num_particles
        self.angles = torch.rand(num_particles, device=device) * 2 * torch.pi
        self.base_speeds = torch.rand(num_particles, device=device) * 0.02 + 0.005
        self.speeds = self.base_speeds * self.speed
        self.distances = torch.zeros(num_particles, device=device)
        
        # Origin (current and target)
        self.origin_x = torch.tensor(initial_origin_x, device=device)
        self.origin_y = torch.tensor(initial_origin_y, device=device)
        self.target_origin_x = self.origin_x.clone()
        self.target_origin_y = self.origin_y.clone()
        
        # Particle-specific inertia factors (will be computed based on distance)
        self.inertia_factors = torch.ones(num_particles, device=device) * inertia_factor
        
        # For respawning particles
        self.active = torch.ones(num_particles, dtype=torch.bool, device=device)
        
        # Frame counters
        self.frame_count = 0
        self.last_major_refresh = 0

    def update(self, 
               new_origin_x: float = None, 
               new_origin_y: float = None, 
               particle_size: float = None,
               speed: float = None,
               inertia_factor: float = None,
               distance_response: float = None) -> torch.Tensor:
        """
        Update particle positions and generate next depth map frame with inertial origin movement.
        
        Args:
            new_origin_x (float, optional): Target origin X (0-1)
            new_origin_y (float, optional): Target origin Y (0-1)
            particle_size (float, optional): Update particle size
            speed (float, optional): Update overall movement speed
            inertia_factor (float, optional): Update inertia factor
            distance_response (float, optional): Update distance response factor
        
        Returns:
            torch.Tensor: Depth map (B, H, W, 3)
        """
        self.frame_count += 1
        
        # Update target origin if provided
        if new_origin_x is not None:
            self.target_origin_x = torch.tensor(new_origin_x, device=self.device)
        if new_origin_y is not None:
            self.target_origin_y = torch.tensor(new_origin_y, device=self.device)
        if particle_size is not None:
            self.particle_size = particle_size
        if speed is not None:
            self.speed = speed
            # Update actual speeds based on base speeds and speed multiplier
            self.speeds = self.base_speeds * self.speed
        if inertia_factor is not None:
            self.base_inertia_factor = inertia_factor
        if distance_response is not None:
            self.distance_response = distance_response

        # Periodic randomization
        if self.frame_count - self.last_major_refresh >= 200:
            refresh_mask = torch.rand(self.num_particles, device=self.device) < 0.3
            if refresh_mask.any():
                num_refresh = refresh_mask.sum()
                self.distances[refresh_mask] = torch.rand(num_refresh, device=self.device) * 0.3
                self.angles[refresh_mask] = torch.rand(num_refresh, device=self.device) * 2 * torch.pi
                self.base_speeds[refresh_mask] = torch.rand(num_refresh, device=self.device) * 0.02 + 0.005
                self.speeds[refresh_mask] = self.base_speeds[refresh_mask] * self.speed
            
            speed_adjust = (torch.rand(self.num_particles, device=self.device) * 0.01) - 0.005
            self.base_speeds = torch.clamp(self.base_speeds + speed_adjust, 0.002, 0.03)
            self.speeds = self.base_speeds * self.speed
            self.last_major_refresh = self.frame_count

        # Update distances
        self.distances = self.distances + self.speeds
        
        # Respawn off-screen particles
        off_screen = self.distances > 1.414
        if off_screen.any():
            num_respawn = off_screen.sum()
            self.distances[off_screen] = torch.rand(num_respawn, device=self.device) * 0.1
            self.angles[off_screen] = torch.rand(num_respawn, device=self.device) * 2 * torch.pi
            self.speeds[off_screen] = torch.rand(num_respawn, device=self.device) * 0.02 + 0.005

        # Calculate current particle positions (before origin update)
        particle_x = self.origin_x + torch.cos(self.angles) * self.distances
        particle_y = self.origin_y + torch.sin(self.angles) * self.distances
        
        # Compute distance-based inertia factors
        particle_distances = torch.sqrt((particle_x - self.origin_x)**2 + (particle_y - self.origin_y)**2)
        
        # Calculate inertia based on distance and user parameters
        # When distance_response is positive, closer particles respond faster
        # When negative, distant particles respond faster
        # When zero, all particles respond uniformly
        base_response = self.base_inertia_factor * 2
        distance_effect = particle_distances * self.distance_response * self.base_inertia_factor
        
        if self.distance_response >= 0:
            # Positive: closer particles respond faster (subtract distance effect)
            self.inertia_factors = torch.clamp(
                base_response - distance_effect,
                0.02 * self.base_inertia_factor, 
                0.2 * self.base_inertia_factor
            )
        else:
            # Negative: distant particles respond faster (add distance effect)
            self.inertia_factors = torch.clamp(
                base_response + distance_effect,
                0.02 * self.base_inertia_factor, 
                0.2 * self.base_inertia_factor
            )

        # Interpolate origin toward target with inertia
        self.origin_x = self.origin_x + self.inertia_factors * (self.target_origin_x - self.origin_x)
        self.origin_y = self.origin_y + self.inertia_factors * (self.target_origin_y - self.origin_y)
        
        # Recalculate particle positions with updated origin
        particle_x = self.origin_x + torch.cos(self.angles) * self.distances
        particle_y = self.origin_y + torch.sin(self.angles) * self.distances

        # Generate depth map
        depth = torch.zeros(self.shape, device=self.device)
        for i in range(self.num_particles):
            if not self.active[i]:
                continue
                
            dist = torch.sqrt((self.x_grid - particle_x[i])**2 + 
                            (self.y_grid - particle_y[i])**2)
            particle = torch.exp(-(dist**2) / (2 * self.particle_size**2))
            depth[0] = torch.maximum(depth[0], particle * self.distances[i])

        depth = depth / (depth.max() + 1e-6)
        return depth.unsqueeze(-1).expand(-1, -1, -1, 3).clamp(0, 1)
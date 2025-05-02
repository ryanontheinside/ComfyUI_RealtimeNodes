"""
Transform control nodes for manipulating and creating transformation matrices.

These nodes allow for precise control of affine transformations in real-time.
"""

import torch
import math

from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.utils.transforms import (
    create_identity_matrix,
    create_translation_matrix,
    create_rotation_matrix,
    create_scale_matrix,
    compose_transforms
)


class TransformMatrixNode(ControlNodeBase):
    """
    Node that maintains and manipulates an affine transformation matrix with precise control.
    
    Allows for precise mathematical operations on transformation matrices.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "operation": (
                ["initialize", "translate", "rotate", "scale", "compose", "reset"], 
                {"default": "initialize"}
            ),
            "translate_x": (
                "FLOAT", 
                {"default": 0.0, "step": 0.01, "tooltip": "Translation in x direction (-1 to 1 range)"}
            ),
            "translate_y": (
                "FLOAT", 
                {"default": 0.0, "step": 0.01, "tooltip": "Translation in y direction (-1 to 1 range)"}
            ),
            "rotate_angle": (
                "FLOAT", 
                {"default": 0.0, "step": 0.1, "tooltip": "Rotation angle in degrees"}
            ),
            "scale_factor_x": (
                "FLOAT", 
                {"default": 1.0, "step": 0.01, "tooltip": "Scale factor in x direction"}
            ),
            "scale_factor_y": (
                "FLOAT", 
                {"default": 1.0, "step": 0.01, "tooltip": "Scale factor in y direction"}
            ),
            "center_x": (
                "FLOAT", 
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, 
                 "tooltip": "X coordinate of transformation center (0-1 range)"}
            ),
            "center_y": (
                "FLOAT", 
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                 "tooltip": "Y coordinate of transformation center (0-1 range)"}
            ),
        })
        inputs["optional"] = {
            "secondary_matrix": (
                "TRANSFORM_MATRIX", 
                {"tooltip": "Secondary matrix for composition operations"}
            )
        }
        return inputs
        
    RETURN_TYPES = ("TRANSFORM_MATRIX",)
    RETURN_NAMES = ("transform_matrix",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/transform"
    
    def update(
        self, 
        operation, 
        translate_x, 
        translate_y, 
        rotate_angle, 
        scale_factor_x, 
        scale_factor_y, 
        center_x, 
        center_y, 
        secondary_matrix=None,
        always_execute=True, 
        unique_id=None
    ):
        """
        Update the transformation matrix based on the specified operation.
        
        Args:
            operation: The operation to perform on the matrix
            translate_x: X translation amount
            translate_y: Y translation amount
            rotate_angle: Rotation angle in degrees
            scale_factor_x: X scale factor
            scale_factor_y: Y scale factor
            center_x: X coordinate of transformation center
            center_y: Y coordinate of transformation center
            secondary_matrix: Optional secondary matrix for composition
            always_execute: Whether to always execute (from ControlNodeBase)
            unique_id: Unique ID for state management (from ControlNodeBase)
            
        Returns:
            Updated transformation matrix
        """
        # Get state or create identity matrix
        state = self.get_state({"matrix": None}, unique_id)
        
        # Default to CPU if we don't have a device yet
        device = "cpu"
        if state["matrix"] is not None:
            device = state["matrix"].device
        elif secondary_matrix is not None:
            device = secondary_matrix.device
            
        current_matrix = state["matrix"] if state["matrix"] is not None else create_identity_matrix(device)
        
        # Perform operation
        if operation == "initialize":
            current_matrix = create_identity_matrix(device)
            
        elif operation == "translate":
            t_matrix = create_translation_matrix(translate_x, translate_y, device)
            current_matrix = compose_transforms(t_matrix, current_matrix)
            
        elif operation == "rotate":
            r_matrix = create_rotation_matrix(rotate_angle, center_x, center_y, device)
            current_matrix = compose_transforms(r_matrix, current_matrix)
            
        elif operation == "scale":
            s_matrix = create_scale_matrix(scale_factor_x, scale_factor_y, center_x, center_y, device)
            current_matrix = compose_transforms(s_matrix, current_matrix)
            
        elif operation == "compose" and secondary_matrix is not None:
            current_matrix = compose_transforms(secondary_matrix, current_matrix)
            
        elif operation == "reset":
            current_matrix = create_identity_matrix(device)
            
        # Update state
        state["matrix"] = current_matrix
        self.set_state(state, unique_id)
        
        return (current_matrix,)


class CameraControlNode(ControlNodeBase):
    """
    High-level camera control with intuitive, independent parameters.
    
    Provides artist-friendly controls for camera manipulation in real-time.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "pan_x": (
                "FLOAT", 
                {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                 "tooltip": "Horizontal panning (-1.0 to 1.0)"}
            ),
            "pan_y": (
                "FLOAT", 
                {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                 "tooltip": "Vertical panning (-1.0 to 1.0)"}
            ),
            "zoom": (
                "FLOAT", 
                {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01,
                 "tooltip": "Zoom factor (1.0 = no zoom)"}
            ),
            "rotation": (
                "FLOAT", 
                {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1,
                 "tooltip": "Rotation in degrees"}
            ),
            "center_x": (
                "FLOAT", 
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                 "tooltip": "X coordinate of rotation/zoom center"}
            ),
            "center_y": (
                "FLOAT", 
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                 "tooltip": "Y coordinate of rotation/zoom center"}
            ),
            "reset": (
                "BOOLEAN", 
                {"default": False, "tooltip": "Reset camera to default state"}
            ),
            "accumulate": (
                "BOOLEAN", 
                {"default": False, "tooltip": "Accumulate transformations over time (vs. absolute values)"}
            ),
        })
        inputs["optional"] = {
            "base_matrix": (
                "TRANSFORM_MATRIX", 
                {"tooltip": "Starting transform matrix (optional)"}
            )
        }
        return inputs
        
    RETURN_TYPES = ("TRANSFORM_MATRIX",)
    RETURN_NAMES = ("transform_matrix",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/transform"
    
    def update(
        self, 
        pan_x, 
        pan_y, 
        zoom, 
        rotation, 
        center_x, 
        center_y,
        reset,
        accumulate,
        base_matrix=None, 
        always_execute=True, 
        unique_id=None
    ):
        """
        Update the camera transformation matrix.
        
        Args:
            pan_x: Horizontal panning amount
            pan_y: Vertical panning amount
            zoom: Zoom factor
            rotation: Rotation angle in degrees
            center_x: X coordinate of rotation/zoom center
            center_y: Y coordinate of rotation/zoom center
            reset: Whether to reset the camera
            accumulate: Whether to accumulate transforms (vs. absolute values)
            base_matrix: Optional base matrix to start from
            always_execute: Whether to always execute (from ControlNodeBase)
            unique_id: Unique ID for state management (from ControlNodeBase)
            
        Returns:
            Updated camera transformation matrix
        """
        # Get state or initialize
        state = self.get_state({
            "matrix": None,  # Current transform matrix
            "last_params": {  # Last used parameters
                "pan_x": 0.0,
                "pan_y": 0.0,
                "zoom": 1.0,
                "rotation": 0.0
            }
        }, unique_id)
        
        device = "cpu"
        
        # Handle reset
        if reset:
            state["matrix"] = None
            state["last_params"] = {
                "pan_x": 0.0,
                "pan_y": 0.0,
                "zoom": 1.0,
                "rotation": 0.0
            }
        
        # Start with base_matrix or identity
        if base_matrix is not None:
            current_matrix = base_matrix
            device = base_matrix.device
        elif not accumulate:
            # For non-accumulate mode, always start fresh
            current_matrix = create_identity_matrix(device)
        elif state["matrix"] is not None:
            current_matrix = state["matrix"]
            device = current_matrix.device
        else:
            current_matrix = create_identity_matrix(device)
        
        # For accumulate mode, calculate the delta from last parameters
        if accumulate and state["last_params"] is not None:
            # Calculate deltas
            delta_pan_x = pan_x - state["last_params"]["pan_x"]
            delta_pan_y = pan_y - state["last_params"]["pan_y"]
            delta_zoom = zoom / state["last_params"]["zoom"] if state["last_params"]["zoom"] != 0 else zoom
            delta_rotation = rotation - state["last_params"]["rotation"]
            
            # Use deltas instead of absolute values
            pan_x = delta_pan_x
            pan_y = delta_pan_y
            zoom = delta_zoom
            rotation = delta_rotation
        
        # Apply transforms in sequence: first scale (zoom), then rotate, then translate (pan)
        # This order ensures consistent behavior
        
        # 1. Apply zoom
        if zoom != 1.0:
            zoom_matrix = create_scale_matrix(zoom, zoom, center_x, center_y, device)
            current_matrix = compose_transforms(zoom_matrix, current_matrix)
        
        # 2. Apply rotation
        if rotation != 0.0:
            rotation_matrix = create_rotation_matrix(rotation, center_x, center_y, device)
            current_matrix = compose_transforms(rotation_matrix, current_matrix)
        
        # 3. Apply panning
        if pan_x != 0.0 or pan_y != 0.0:
            translation_matrix = create_translation_matrix(pan_x, pan_y, device)
            current_matrix = compose_transforms(translation_matrix, current_matrix)
        
        # Update state with current parameters and matrix
        state["matrix"] = current_matrix
        state["last_params"] = {
            "pan_x": pan_x if not accumulate else pan_x + state["last_params"]["pan_x"],
            "pan_y": pan_y if not accumulate else pan_y + state["last_params"]["pan_y"],
            "zoom": zoom if not accumulate else zoom * state["last_params"]["zoom"],
            "rotation": rotation if not accumulate else rotation + state["last_params"]["rotation"]
        }
        self.set_state(state, unique_id)
        
        return (current_matrix,) 
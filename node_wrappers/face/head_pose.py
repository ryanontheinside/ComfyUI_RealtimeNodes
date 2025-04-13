# node_wrappers/face/head_pose.py
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Any, Tuple, Dict

from ...src.types import TRANSFORM_MATRIX_LIST
from ...src.utils.delta_utils import scale_value
from ..common.base_delta_nodes import FLOAT_EQUALITY_TOLERANCE # For trigger node comparisons

logger = logging.getLogger(__name__)

_category = "MediaPipeVision/Face/HeadPose"

# --- Helper Functions ---
def get_first_transform_matrix(matrix_result: TRANSFORM_MATRIX_LIST) -> Optional[np.ndarray]:
    """Extracts the first 4x4 transformation matrix from the result."""
    if matrix_result is None or len(matrix_result) == 0:
        return None
    if len(matrix_result[0]) == 0:
        return None
    if matrix_result[0][0] is None:
        return None
    # FACE_TRANSFORM_MATRICES structure: List[List[np.ndarray]]
    return matrix_result[0][0]

def decompose_transform_matrix(matrix: np.ndarray) -> Optional[Dict[str, float]]:
    """Decomposes a 4x4 matrix into translation (x, y, z) and Euler angles (pitch, yaw, roll)."""
    if matrix is None or matrix.shape != (4, 4):
        return None
    try:
        # Extract translation
        translation = matrix[:3, 3]
        # Extract rotation matrix part
        rotation_matrix = matrix[:3, :3]
        # Convert rotation matrix to Euler angles (ZYX convention common for head pose)
        # Note: Scipy uses intrinsic rotations. ZYX order corresponds to yaw, pitch, roll.
        rotation = R.from_matrix(rotation_matrix)
        euler_angles_rad = rotation.as_euler('zyx', degrees=False)
        # Convert radians to degrees for easier interpretation
        euler_angles_deg = np.degrees(euler_angles_rad)
        
        return {
            "x": translation[0],
            "y": translation[1],
            "z": translation[2],
            "yaw": euler_angles_deg[0],   # Rotation around Y (vertical) axis
            "pitch": euler_angles_deg[1], # Rotation around X (side-to-side) axis
            "roll": euler_angles_deg[2],  # Rotation around Z (front-to-back) axis
        }
    except Exception as e:
        logger.error(f"Error decomposing matrix: {e}")
        return None


# --- Control Node ---
HEAD_POSE_COMPONENTS = ["x", "y", "z", "pitch", "yaw", "roll"]

class HeadPoseControlFloatNode:
    """Outputs a scaled FLOAT based on a selected component (translation/rotation) of the head pose."""
    CATEGORY = _category
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    DESCRIPTION = "Converts head position and orientation into floating point values. Use this to control parameters based on head movements like nodding, turning left/right, or tilting."

    @classmethod
    def INPUT_TYPES(cls):
        # Example ranges - these might need tuning based on expected matrix values
        default_ranges = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-1.0, 0.5), # Example translation ranges
            "pitch": (-45.0, 45.0), "yaw": (-60.0, 60.0), "roll": (-45.0, 45.0) # Example angle ranges (degrees)
        }
        default_component = "yaw"
        default_min, default_max = default_ranges[default_component]
        
        return {
            "required": {
                "face_transform_matrices": ("TRANSFORM_MATRIX_LIST", {"tooltip": "3D head orientation data from the Face Landmarker node"}),
                "component": (HEAD_POSE_COMPONENTS, {"default": default_component, 
                                                   "tooltip": "Which aspect of head movement to track - yaw (left/right), pitch (up/down), roll (tilt), or position (x/y/z)"}),
                "value_min": ("FLOAT", {"default": default_min, "step": 0.1, 
                                      "tooltip": "The minimum value of the selected movement component to consider"}),
                "value_max": ("FLOAT", {"default": default_max, "step": 0.1, 
                                      "tooltip": "The maximum value of the selected movement component to consider"}),
                "output_min_float": ("FLOAT", {"default": 0.0, "step": 0.01, 
                                             "tooltip": "The minimum output value when head position is at or below value_min"}),
                "output_max_float": ("FLOAT", {"default": 1.0, "step": 0.01, 
                                             "tooltip": "The maximum output value when head position is at or above value_max"}),
                "clamp": ("BOOLEAN", {"default": True, 
                                    "tooltip": "When enabled, restricts output values to stay within min and max range"}),
            }
        }

    def execute(self, face_transform_matrices, component, value_min, value_max, output_min_float, output_max_float, clamp):
        matrix = get_first_transform_matrix(face_transform_matrices)
        decomposed = decompose_transform_matrix(matrix)
        
        raw_value = None
        if decomposed and component in decomposed:
            raw_value = decomposed[component]
        else:
             logger.debug(f"Could not get head pose component '{component}'")

        scaled_value = scale_value(raw_value, value_min, value_max, output_min_float, output_max_float, clamp)

        # Default to min output value on error
        output_val = scaled_value if scaled_value is not None else float(output_min_float)
        return (output_val,)

class HeadPoseControlIntNode:
    """Outputs a scaled INT based on a selected component (translation/rotation) of the head pose."""
    CATEGORY = _category
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    DESCRIPTION = "Converts head position and orientation into integer values. Perfect for selecting options, changing steps/seeds, or triggering numbered presets based on where someone is looking."

    @classmethod
    def INPUT_TYPES(cls):
        # Share the same default range logic as Float node
        default_ranges = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-1.0, 0.5),
            "pitch": (-45.0, 45.0), "yaw": (-60.0, 60.0), "roll": (-45.0, 45.0)
        }
        default_component = "yaw"
        default_min, default_max = default_ranges[default_component]
        
        return {
            "required": {
                "face_transform_matrices": ("TRANSFORM_MATRIX_LIST", {"tooltip": "3D head orientation data from the Face Landmarker node"}),
                "component": (HEAD_POSE_COMPONENTS, {"default": default_component, 
                                                   "tooltip": "Which aspect of head movement to track - yaw (left/right), pitch (up/down), roll (tilt), or position (x/y/z)"}),
                "value_min": ("FLOAT", {"default": default_min, "step": 0.1, 
                                      "tooltip": "The minimum value of the selected movement component to consider"}),
                "value_max": ("FLOAT", {"default": default_max, "step": 0.1, 
                                      "tooltip": "The maximum value of the selected movement component to consider"}),
                "output_min_int": ("INT", {"default": 0, 
                                         "tooltip": "The minimum integer output value when head position is at or below value_min"}),
                "output_max_int": ("INT", {"default": 100, 
                                         "tooltip": "The maximum integer output value when head position is at or above value_max"}),
                "clamp": ("BOOLEAN", {"default": True, 
                                    "tooltip": "When enabled, restricts output values to stay within min and max range"}),
            }
        }

    def execute(self, face_transform_matrices, component, value_min, value_max, output_min_int, output_max_int, clamp):
        matrix = get_first_transform_matrix(face_transform_matrices)
        decomposed = decompose_transform_matrix(matrix)
        
        raw_value = None
        if decomposed and component in decomposed:
            raw_value = decomposed[component]
        else:
            logger.debug(f"Could not get head pose component '{component}'")
            
        scaled_value = scale_value(raw_value, value_min, value_max, output_min_int, output_max_int, clamp)

        # Default to min output value on error
        output_val = int(round(scaled_value)) if scaled_value is not None else int(output_min_int)
        return (output_val,)


# --- Trigger Node ---
class HeadPoseTriggerNode:
    """Outputs a BOOLEAN trigger if a head pose component crosses a threshold."""
    CATEGORY = _category
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "execute"
    DESCRIPTION = "Creates a true/false trigger based on head movements exceeding thresholds. Use it to activate workflows when someone nods, looks left/right, or tilts their head a certain amount."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_transform_matrices": ("TRANSFORM_MATRIX_LIST", {"tooltip": "3D head orientation data from the Face Landmarker node"}),
                "component": (HEAD_POSE_COMPONENTS, {"default": "yaw", 
                                                   "tooltip": "Which aspect of head movement to track - yaw (left/right), pitch (up/down), roll (tilt), or position (x/y/z)"}),
                "threshold": ("FLOAT", {"default": 30.0, "step": 0.5, 
                                      "tooltip": "The angle or position value that triggers the condition (example: 30 degrees of head turn)"}),
                "condition": (["Above", "Below", "Equals", "Not Equals"], 
                             {"tooltip": "The condition that determines when to trigger - Above (movement > threshold), Below (movement < threshold), etc."}),
            }
        }

    def execute(self, face_transform_matrices, component, threshold, condition):
        matrix = get_first_transform_matrix(face_transform_matrices)
        decomposed = decompose_transform_matrix(matrix)
        
        raw_value = None
        if decomposed and component in decomposed:
            raw_value = decomposed[component]
        else:
             logger.debug(f"Could not get head pose component '{component}' for trigger")

        triggered = False
        if raw_value is not None:
            if condition == "Above" and raw_value > threshold:
                triggered = True
            elif condition == "Below" and raw_value < threshold:
                triggered = True
            elif condition == "Equals" and abs(raw_value - threshold) < FLOAT_EQUALITY_TOLERANCE:
                triggered = True
            elif condition == "Not Equals" and abs(raw_value - threshold) >= FLOAT_EQUALITY_TOLERANCE:
                triggered = True

        return (triggered,)


# --- Mappings ---
NODE_CLASS_MAPPINGS = {
    "HeadPoseControlFloat": HeadPoseControlFloatNode,
    "HeadPoseControlInt": HeadPoseControlIntNode,
    "HeadPoseTrigger": HeadPoseTriggerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeadPoseControlFloat": "Head Pose Control (Float)",
    "HeadPoseControlInt": "Head Pose Control (Int)",
    "HeadPoseTrigger": "Head Pose Trigger",
} 
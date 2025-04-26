import logging

# Import the base class
from ....src.mediapipe_vision.common.position import LandmarkPositionBase 
# Import the specific landmark type for clarity and potential future use
# Import the landmark definitions
from ....src.mediapipe_vision.landmark_definitions import POSE_LANDMARK_TOOLTIP

logger = logging.getLogger(__name__)
_category = "Realtime Nodes/MediaPipe Vision/Pose/PoseLandmark/Position"

class PoseLandmarkPositionNode(LandmarkPositionBase):
    """Extracts position lists (x, y, z, vis, pres) for a specific landmark index
       from POSE_LANDMARKS across the batch."""
    CATEGORY = _category
    # Define the specific input type string this node expects
    LANDMARKS_TYPE = "POSE_LANDMARKS"
    DESCRIPTION = "Extracts lists of coordinates (x, y, z) and properties (visibility, presence) for a specific Pose Landmark index across an entire batch. Supports world coordinates. Outputs separate lists for each property."

    @classmethod
    def INPUT_TYPES(cls):
        # Define the specific input type for this node
        return {
            "required": {
                "landmarks": (cls.LANDMARKS_TYPE, {"forceInput": True}),
                "landmark_index": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip": POSE_LANDMARK_TOOLTIP}),
                "result_index": ("INT", {"default": 0, "min": 0, "tooltip": "Index of the pose detection to use (0=first detected pose)"}),
                "use_world_coordinates": ("BOOLEAN", {"default": False, "tooltip": "Use world coordinates if available"}),
            }
        }

    # No need to override INPUT_TYPES or extract_positions, inheriting from base is sufficient.
    # The base class handles everything using self.LANDMARKS_TYPE.

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "PoseLandmarkPosition": PoseLandmarkPositionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseLandmarkPosition": "Pose Landmark Position Extractor (Batch)",
} 
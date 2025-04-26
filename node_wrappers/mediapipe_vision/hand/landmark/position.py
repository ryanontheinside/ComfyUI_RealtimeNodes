# node_wrappers/hand/position.py
import logging

# Import the base class
from .....src.mediapipe_vision.common.position import LandmarkPositionBase

# Import the specific landmark type for clarity and potential future use
# Import the landmark definitions
from .....src.mediapipe_vision.landmark_definitions import HAND_LANDMARK_TOOLTIP

logger = logging.getLogger(__name__)
_category = "Realtime Nodes/MediaPipe Vision/Hand/HandLandmark/Position"

class HandLandmarkPositionNode(LandmarkPositionBase):
    """Extracts position lists (x, y, z, vis, pres) for a specific landmark index
       from HAND_LANDMARKS across the batch."""
    CATEGORY = _category
    # Define the specific input type string this node expects
    LANDMARKS_TYPE = "HAND_LANDMARKS"
    DESCRIPTION = "Extracts lists of coordinates (x, y, z) and properties (visibility, presence) for a specific Hand Landmark index across an entire batch. Supports world coordinates. Outputs separate lists for each property."

    @classmethod
    def INPUT_TYPES(cls):
        # Define the specific input type for this node
        return {
            "required": {
                "landmarks": (cls.LANDMARKS_TYPE, {"forceInput": True}),
                "landmark_index": ("INT", {"default": 8, "min": 0, "max": 20, "tooltip": HAND_LANDMARK_TOOLTIP}),
                "result_index": ("INT", {"default": 0, "min": 0, "tooltip": "Index of the hand detection to use (0=first detected hand)"}),
                "use_world_coordinates": ("BOOLEAN", {"default": False, "tooltip": "Use world coordinates if available"}),
            }
        }

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HandLandmarkPosition": HandLandmarkPositionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HandLandmarkPosition": "Hand Landmark Position Extractor (Batch)",
} 
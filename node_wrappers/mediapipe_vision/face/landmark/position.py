import logging
from ...location_landmark.universal_landmark_nodes import LandmarkPositionBaseNode
# Import the landmark definitions
from .....src.mediapipe_vision.landmark_definitions import FACE_LANDMARK_TOOLTIP

logger = logging.getLogger(__name__)

_category = "Realtime Nodes/MediaPipe Vision/Face/FaceLandmark/Position"

class FaceLandmarkPositionNode(LandmarkPositionBaseNode):
    """Extracts position lists (x, y, z, vis, pres) for a specific landmark index
       from FACE_LANDMARKS across the batch."""
    CATEGORY = _category
    # Define the specific input type string this node expects
    LANDMARKS_TYPE = "FACE_LANDMARKS"
    DESCRIPTION = "Extracts lists of coordinates (x, y, z) and properties (visibility, presence) for a specific Face Landmark index across an entire batch. Outputs separate lists for each property."

    @classmethod
    def INPUT_TYPES(cls):
        # Define the specific input type for this node
        return {
            "required": {
                "landmarks": (cls.LANDMARKS_TYPE, {"forceInput": True}), # Use the class variable
                "landmark_index": ("INT", {"default": 0, "min": 0, "tooltip": FACE_LANDMARK_TOOLTIP}),
                "result_index": ("INT", {"default": 0, "min": 0, "tooltip": "Index of the face detection to use (0=first detected face)"}),
            }
        }

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "FaceLandmarkPosition": FaceLandmarkPositionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceLandmarkPosition": "Face Landmark Position Extractor (Batch)",
} 
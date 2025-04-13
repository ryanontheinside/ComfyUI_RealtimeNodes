# node_wrappers/face/delta.py
import logging
from typing import Optional, List, Any

from ...src.types import FACE_LANDMARKS
from ..common.base_delta_nodes import (
    BaseLandmarkDeltaIntControlNode,
    BaseLandmarkDeltaFloatControlNode,
    BaseLandmarkDeltaTriggerNode
)

logger = logging.getLogger(__name__)


_category = "MediaPipeVision/Face/FaceLandmark/Delta"

# --- Helper Function to Extract Face Landmarks --- 
# This function adapts the specific FACE_LANDMARKS structure to the generic list format needed by the base class
def get_face_landmarks_list(landmarks_result: FACE_LANDMARKS) -> Optional[List[Any]]:
    """Extracts the list of Landmark objects from the first detected face in the first batch item."""
    if not landmarks_result or not landmarks_result[0] or not landmarks_result[0][0]:
        return None
    # FACE_LANDMARKS structure: List[List[List[LandmarkPoint]]]
    # We usually care about the first face detected in the first image of the batch.
    return landmarks_result[0][0]


# --- Concrete Nodes --- 
class FaceLandmarkDeltaIntControlNode(BaseLandmarkDeltaIntControlNode):
    """Outputs an INT based on the delta between current and previous face landmarks."""
    CATEGORY = _category
    LANDMARKS_TYPE = "FACE_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 8
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 477
    TOOLTIP = "Index of the landmark to track (e.g., 8=Chin, 0=Nose Tip)"
    DESCRIPTION = "Tracks changes in facial landmark positions and converts movement into integer values. Perfect for creating UI controls that respond to specific parts of the face moving."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_face_landmarks_list(landmarks_result)


class FaceLandmarkDeltaFloatControlNode(BaseLandmarkDeltaFloatControlNode):
    """Outputs a FLOAT based on the delta between current and previous face landmarks."""
    CATEGORY = _category
    LANDMARKS_TYPE = "FACE_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 8
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 477
    TOOLTIP = "Index of the landmark to track (e.g., 8=Chin, 0=Nose Tip)"
    DESCRIPTION = "Tracks changes in facial landmark positions and converts movement into precise float values. Ideal for smooth animation control or gradual parameter adjustments based on facial movement."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_face_landmarks_list(landmarks_result)


class FaceLandmarkDeltaTriggerNode(BaseLandmarkDeltaTriggerNode):
    """Outputs a BOOLEAN trigger if face landmark delta (vs previous) crosses a threshold."""
    CATEGORY = _category
    LANDMARKS_TYPE = "FACE_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 8
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 477
    TOOLTIP = "Index of the landmark to track (e.g., 8=Chin, 0=Nose Tip)"
    DESCRIPTION = "Creates true/false triggers when specific parts of the face move beyond a threshold. Great for detecting sudden facial movements like mouth opening, blinking, or eyebrow raises."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_face_landmarks_list(landmarks_result)


# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "FaceLandmarkDeltaIntControl": FaceLandmarkDeltaIntControlNode,
    "FaceLandmarkDeltaFloatControl": FaceLandmarkDeltaFloatControlNode,
    "FaceLandmarkDeltaTrigger": FaceLandmarkDeltaTriggerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceLandmarkDeltaIntControl": "Face Landmark Delta Int Control",
    "FaceLandmarkDeltaFloatControl": "Face Landmark Delta Float Control",
    "FaceLandmarkDeltaTrigger": "Face Landmark Delta Trigger",
} 
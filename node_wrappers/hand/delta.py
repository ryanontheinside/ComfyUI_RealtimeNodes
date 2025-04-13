# node_wrappers/hand/delta.py
import logging
from typing import Optional, List, Any

from ...src.types import HAND_LANDMARKS, HandLandmarksResult
from ..common.base_delta_nodes import (
    BaseLandmarkDeltaIntControlNode,
    BaseLandmarkDeltaFloatControlNode,
    BaseLandmarkDeltaTriggerNode
)

logger = logging.getLogger(__name__)

_category = "MediaPipeVision/Hand/HandLandmark/Delta"
# --- Helper Function to Extract Hand Landmarks --- 
def get_hand_landmarks_list(landmarks_result: HAND_LANDMARKS) -> Optional[List[Any]]:
    """Extracts the list of Landmark objects from the first detected hand in the first batch item."""
    if not landmarks_result or not landmarks_result[0] or not landmarks_result[0][0]:
        return None
    # HAND_LANDMARKS structure: List[List[HandLandmarksResult]]
    # HandLandmarksResult contains the 'landmarks' list.
    first_hand_result: HandLandmarksResult = landmarks_result[0][0]
    return first_hand_result.landmarks


# --- Concrete Nodes --- 
class HandLandmarkDeltaIntControlNode(BaseLandmarkDeltaIntControlNode):
    """Outputs an INT based on the delta between current and previous hand landmarks."""
    CATEGORY = _category
    LANDMARKS_TYPE = "HAND_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 8
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 20
    TOOLTIP = "Index Tip=8, Thumb Tip=4"
    DESCRIPTION = "Tracks changes in hand landmark positions and converts movement into integer values. Perfect for creating UI controls that respond to finger or hand movement."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_hand_landmarks_list(landmarks_result)


class HandLandmarkDeltaFloatControlNode(BaseLandmarkDeltaFloatControlNode):
    """Outputs a FLOAT based on the delta between current and previous hand landmarks."""
    CATEGORY = _category
    LANDMARKS_TYPE = "HAND_LANDMARKS" 
    DEFAULT_LANDMARK_INDEX = 8
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 20
    TOOLTIP = "Index Tip=8, Thumb Tip=4"
    DESCRIPTION = "Tracks changes in hand landmark positions and converts movement into precise float values. Ideal for smooth animation control or gradual parameter adjustments based on finger movement."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_hand_landmarks_list(landmarks_result)


class HandLandmarkDeltaTriggerNode(BaseLandmarkDeltaTriggerNode):
    """Outputs a BOOLEAN trigger if hand landmark delta (vs previous) crosses a threshold."""
    CATEGORY = _category
    LANDMARKS_TYPE = "HAND_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 8
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 20
    TOOLTIP = "Index Tip=8, Thumb Tip=4"
    DESCRIPTION = "Creates true/false triggers when specific parts of the hand move beyond a threshold. Great for detecting sudden finger movements, hand gestures, or tracking motion for interactive applications."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_hand_landmarks_list(landmarks_result)


# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "HandLandmarkDeltaIntControl": HandLandmarkDeltaIntControlNode,
    "HandLandmarkDeltaFloatControl": HandLandmarkDeltaFloatControlNode,
    "HandLandmarkDeltaTrigger": HandLandmarkDeltaTriggerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HandLandmarkDeltaIntControl": "Hand Landmark Delta Int Control",
    "HandLandmarkDeltaFloatControl": "Hand Landmark Delta Float Control",
    "HandLandmarkDeltaTrigger": "Hand Landmark Delta Trigger",
} 
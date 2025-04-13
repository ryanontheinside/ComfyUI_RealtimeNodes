# node_wrappers/pose/delta.py
import logging
from typing import Optional, List, Any

from ...src.types import POSE_LANDMARKS, PoseLandmarksResult
from ..common.base_delta_nodes import (
    BaseLandmarkDeltaIntControlNode,
    BaseLandmarkDeltaFloatControlNode,
    BaseLandmarkDeltaTriggerNode
)

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Pose/Delta"

# --- Helper Function to Extract Pose Landmarks --- 
def get_pose_landmarks_list(landmarks_result: POSE_LANDMARKS) -> Optional[List[Any]]:
    """Extracts the list of Landmark objects from the first detected pose in the first batch item."""
    if not landmarks_result or not landmarks_result[0] or not landmarks_result[0][0]:
        return None
    # POSE_LANDMARKS structure: List[List[PoseLandmarksResult]]
    # PoseLandmarksResult contains the 'landmarks' list.
    first_pose_result: PoseLandmarksResult = landmarks_result[0][0]
    return first_pose_result.landmarks


# --- Concrete Nodes --- 
class PoseLandmarkDeltaIntControlNode(BaseLandmarkDeltaIntControlNode):
    """Outputs an INT based on the delta between current and previous pose landmarks."""
    CATEGORY = _category
    LANDMARKS_TYPE = "POSE_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 0
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 32
    TOOLTIP = "Nose=0, Left Wrist=15, Right Wrist=16"
    DESCRIPTION = "Tracks changes in body position and converts movement into integer values. Perfect for creating UI controls that respond to specific body parts moving."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_pose_landmarks_list(landmarks_result)


class PoseLandmarkDeltaFloatControlNode(BaseLandmarkDeltaFloatControlNode):
    """Outputs a FLOAT based on the delta between current and previous pose landmarks."""
    CATEGORY = _category
    LANDMARKS_TYPE = "POSE_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 0
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 32
    TOOLTIP = "Nose=0, Left Wrist=15, Right Wrist=16"
    DESCRIPTION = "Tracks changes in body position and converts movement into precise float values. Ideal for smooth animation control or gradual parameter adjustments based on body movement."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_pose_landmarks_list(landmarks_result)


class PoseLandmarkDeltaTriggerNode(BaseLandmarkDeltaTriggerNode):
    """Outputs a BOOLEAN trigger if pose landmark delta (vs previous) crosses a threshold."""
    CATEGORY = _category
    LANDMARKS_TYPE = "POSE_LANDMARKS"
    DEFAULT_LANDMARK_INDEX = 0
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 32
    TOOLTIP = "Nose=0, Left Wrist=15, Right Wrist=16"
    DESCRIPTION = "Creates true/false triggers when specific body parts move beyond a threshold. Great for detecting gestures, dance moves, or tracking significant changes in body position."
    
    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        return get_pose_landmarks_list(landmarks_result)


# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "PoseLandmarkDeltaIntControl": PoseLandmarkDeltaIntControlNode,
    "PoseLandmarkDeltaFloatControl": PoseLandmarkDeltaFloatControlNode,
    "PoseLandmarkDeltaTrigger": PoseLandmarkDeltaTriggerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseLandmarkDeltaIntControl": "Pose Landmark Delta Int Control",
    "PoseLandmarkDeltaFloatControl": "Pose Landmark Delta Float Control",
    "PoseLandmarkDeltaTrigger": "Pose Landmark Delta Trigger",
} 
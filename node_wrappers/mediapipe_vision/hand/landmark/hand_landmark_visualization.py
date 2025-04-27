# visualize_hand.py
import logging
from typing import Any, List, Optional

import mediapipe as mp

from .....src.mediapipe_vision.common.base_visualization_nodes import BaseLandmarkVisualizationNode

logger = logging.getLogger(__name__)
_category = "Realtime Nodes/MediaPipe Vision/Hand/HandLandmark/Visualization"
# Define standard hand connections
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


class VisualizeHandLandmarks(BaseLandmarkVisualizationNode):
    """Draws Hand landmarks and connections onto images."""

    # Override base class properties
    LANDMARKS_TYPE = "HAND_LANDMARKS"
    CONNECTIONS = HAND_CONNECTIONS
    DEFAULT_DRAW_POINTS = True
    DEFAULT_POINT_COLOR = "#FF0000"  # Red
    DEFAULT_POINT_RADIUS = 3
    DEFAULT_CONNECTION_COLOR = "#00FF00"  # Green
    DEFAULT_CONNECTION_THICKNESS = 2
    SUPPORTS_VISIBILITY = False
    CATEGORY = _category
    DESCRIPTION = "Visualizes hand landmarks by drawing points and connections on detected hands. Creates a wireframe overlay showing finger joints, palm, and hand structure."

    def get_landmark_list(self, detection_result: Any) -> Optional[List[Any]]:
        return detection_result.landmarks


# Node registration
NODE_CLASS_MAPPINGS = {"VisualizeHandLandmarks": VisualizeHandLandmarks}

NODE_DISPLAY_NAME_MAPPINGS = {"VisualizeHandLandmarks": "Visualize Hand Landmarks (MediaPipe)"}

from typing import Any, List

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider  # Import TimestampProvider
from ..common import BaseDetector
from ..types import GestureCategory, GestureRecognitionResult


class GestureRecognizer(BaseDetector[GestureRecognitionResult]):
    """Recognizes hand gestures using MediaPipe GestureRecognizer."""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = None  # Added

    def _create_detector_options(self, base_options: python.BaseOptions, 
                               mode_enum: vision.RunningMode, **kwargs) -> vision.GestureRecognizerOptions:
        """Create GestureRecognizer-specific options with parameters:
            - num_hands: Maximum number of hands to detect
            - min_detection_confidence: Minimum confidence for hand detection
            - min_tracking_confidence: Minimum confidence for hand tracking
            - min_presence_confidence: Minimum confidence for hand presence
        """
        return vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            num_hands=kwargs.get('num_hands', 2),
            min_hand_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            min_hand_presence_confidence=kwargs.get('min_presence_confidence', 0.5),
        )

    def _create_detector_instance(self, options: vision.GestureRecognizerOptions) -> vision.GestureRecognizer:
        """Create GestureRecognizer instance.
        
        Args:
            options: Configured GestureRecognizerOptions.
            
        Returns:
            MediaPipe GestureRecognizer instance.
        """
        return vision.GestureRecognizer.create_from_options(options)

    def _get_options_tuple(self, running_mode: str = None, delegate: str = None, **kwargs) -> tuple:
        """Get tuple of options for comparison.
        
        Args:
            running_mode: String representation of running mode ("image" or "video").
            delegate: String representation of delegate ("cpu" or "gpu").
            **kwargs: GestureRecognizer-specific parameters.
            
        Returns:
            Tuple containing all configuration options for comparison.
        """
        if self._current_options:
            # Getting values from existing options object
            return (
                self._current_options.num_hands,
                self._current_options.min_hand_detection_confidence,
                self._current_options.min_tracking_confidence,
                self._current_options.min_hand_presence_confidence,
                running_mode,
                delegate,
            )
        else:
            # Getting values from kwargs for new options
            return (
                kwargs.get('num_hands', 2),
                kwargs.get('min_detection_confidence', 0.5),
                kwargs.get('min_tracking_confidence', 0.5),
                kwargs.get('min_presence_confidence', 0.5),
                running_mode,
                delegate,
            )

    def _process_detection_result(self, recognition_result: Any) -> List[GestureRecognitionResult]:
        """Process GestureRecognizer detection results.
        
        Args:
            recognition_result: Raw GestureRecognizer result from MediaPipe.
            
        Returns:
            List of GestureRecognitionResult objects.
        """
        current_image_gestures = []
        if recognition_result and recognition_result.gestures:
            for hand_idx, hand_gestures in enumerate(recognition_result.gestures):
                gesture_categories = [
                    GestureCategory(
                        index=cat.index,
                        score=cat.score,
                        display_name=cat.display_name,
                        category_name=cat.category_name,
                    )
                    for cat in hand_gestures
                ]
                handedness = None
                if recognition_result.handedness and hand_idx < len(recognition_result.handedness):
                    handedness_cats = recognition_result.handedness[hand_idx]
                    if handedness_cats:
                        handedness = handedness_cats[0].display_name
                current_image_gestures.append(GestureRecognitionResult(gestures=gesture_categories, handedness=handedness))
        return current_image_gestures

    # Public API method that keeps the same signature as before
    def recognize(
        self,
        image: torch.Tensor,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        running_mode: str = "video",
        delegate: str = "cpu",
    ) -> List[List[GestureRecognitionResult]]:
        """Recognizes hand gestures in the input image tensor."""
        return self.detect(
            image,
            running_mode=running_mode,
            delegate=delegate,
            num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_presence_confidence=min_presence_confidence,
        )

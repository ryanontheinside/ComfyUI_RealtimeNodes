from typing import Any, List

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider  # Import TimestampProvider
from ..common import BaseDetector
from ..types import HandLandmarksResult, LandmarkPoint


class HandLandmarkDetector(BaseDetector[HandLandmarksResult]):
    """Detects hand landmarks using MediaPipe HandLandmarker."""

    def __init__(self, model_path: str):
        """Initialize the detector with the model path.

        Args:
            model_path: Path to the MediaPipe HandLandmarker .task file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        # Verify file exists? Optional, MediaPipe might handle this.

        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = None  # Added

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.HandLandmarkerOptions:
        """Create HandLandmarker-specific options with parameters:
            - num_hands: Maximum number of hands to detect
            - min_detection_confidence: Minimum confidence for hand detection
            - min_presence_confidence: Minimum confidence for hand presence
            - min_tracking_confidence: Minimum confidence for hand tracking
        """
        return vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            num_hands=kwargs.get('num_hands', 2),
            min_hand_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
            min_hand_presence_confidence=kwargs.get('min_presence_confidence', 0.5),
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
        )

    def _create_detector_instance(self, options: vision.HandLandmarkerOptions) -> vision.HandLandmarker:
        return vision.HandLandmarker.create_from_options(options)

    def _get_options_tuple(self, running_mode: str = None, delegate: str = None, **kwargs) -> tuple:
        if self._current_options:
            return (
                self._current_options.num_hands,
                self._current_options.min_hand_detection_confidence,
                self._current_options.min_hand_presence_confidence,
                self._current_options.min_tracking_confidence,
                running_mode,
                delegate,
            )
        else:
            return (
                kwargs.get('num_hands', 2),
                kwargs.get('min_detection_confidence', 0.5),
                kwargs.get('min_presence_confidence', 0.5),
                kwargs.get('min_tracking_confidence', 0.5),
                running_mode,
                delegate,
            )

    def _process_detection_result(self, detection_result: Any) -> List[HandLandmarksResult]:
        current_image_results = []
        if detection_result and detection_result.hand_landmarks:
            for hand_idx, hand_landmarks_mp in enumerate(detection_result.hand_landmarks):
                landmarks = [LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z) for lm_idx, lm in enumerate(hand_landmarks_mp)]

                world_landmarks = None
                if detection_result.hand_world_landmarks and hand_idx < len(detection_result.hand_world_landmarks):
                    world_landmarks = [
                        LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z)
                        for lm_idx, lm in enumerate(detection_result.hand_world_landmarks[hand_idx])
                    ]

                handedness = None
                if detection_result.handedness and hand_idx < len(detection_result.handedness):
                    handedness_cats = detection_result.handedness[hand_idx]
                    if handedness_cats:
                        handedness = handedness_cats[0].display_name

                current_image_results.append(
                    HandLandmarksResult(landmarks=landmarks, world_landmarks=world_landmarks, handedness=handedness)
                )
        return current_image_results

    def detect(
        self,
        image: torch.Tensor,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "video",
        delegate: str = "cpu",
    ) -> List[List[HandLandmarksResult]]:
        """Detects hand landmarks in the input image tensor."""
        return super().detect(
            image,
            running_mode=running_mode,
            delegate=delegate,
            num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

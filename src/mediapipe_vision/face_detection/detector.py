"""Face Detection implementation using MediaPipe.

This module contains the implementation of MediaPipe Face Detection functionality.
"""

from typing import Any, List

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider  # Import from utils
from ..common import BaseDetector
from ..types import BoundingBox, FaceDetectionResult, FaceKeypoint


class FaceDetector(BaseDetector[FaceDetectionResult]):
    """Detects faces in an image using MediaPipe FaceDetector."""

    def __init__(self, model_path: str):
        """Initialize the detector with the model path.

        Args:
            model_path: Path to the MediaPipe FaceDetector .task file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")

        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None  # Store current options for comparison
        self._timestamp_provider = None  # Initialize TimestampProvider later

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.FaceDetectorOptions:
        """Create FaceDetector-specific options with parameters:
            - min_detection_confidence: Minimum confidence for face detection
        """
        return vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mode_enum,
            min_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
        )

    def _create_detector_instance(self, options: vision.FaceDetectorOptions) -> vision.FaceDetector:
        return vision.FaceDetector.create_from_options(options)

    def _get_options_tuple(self, running_mode: str = None, delegate: str = None, **kwargs) -> tuple:
        if self._current_options:
            return (
                self._current_options.min_detection_confidence,
                running_mode,
                delegate,
            )
        else:
            return (
                kwargs.get('min_detection_confidence', 0.5),
                running_mode,
                delegate,
            )

    def _process_detection_result(self, detection_result: Any) -> List[FaceDetectionResult]:
        current_image_detections = []
        if detection_result and detection_result.detections:
            for detection in detection_result.detections:
                bbox_mp = detection.bounding_box
                keypoints_mp = detection.keypoints
                score = detection.categories[0].score if detection.categories else None

                bbox = BoundingBox(origin_x=bbox_mp.origin_x, origin_y=bbox_mp.origin_y, width=bbox_mp.width, height=bbox_mp.height)

                keypoints = None
                if keypoints_mp:
                    keypoints = [FaceKeypoint(label=kp.label, x=kp.x, y=kp.y) for kp in keypoints_mp]

                current_image_detections.append(FaceDetectionResult(bounding_box=bbox, keypoints=keypoints, score=score))
        return current_image_detections

    def detect(
        self,
        image: torch.Tensor,
        min_detection_confidence: float = 0.5,
        running_mode: str = "video",
        delegate: str = "cpu",
    ) -> List[List[FaceDetectionResult]]:
        """Detects faces in the input image tensor."""
        return super().detect(
            image,
            running_mode=running_mode,
            delegate=delegate,
            min_detection_confidence=min_detection_confidence,
        )

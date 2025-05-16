from typing import Any, List

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider
from ..common import BaseDetector
from ..types import BoundingBox, ObjectDetectionCategory, ObjectDetectionResult


class ObjectDetector(BaseDetector[ObjectDetectionResult]):
    """Detects objects using MediaPipe ObjectDetector."""

    def __init__(self, model_path: str):
        """Initialize the detector with the model path.

        Args:
            model_path: Path to the MediaPipe ObjectDetector .tflite file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")

        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = None  # Added

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.ObjectDetectorOptions:
        """Create ObjectDetector-specific options with parameters:
            - score_threshold: Minimum score for object detection
            - max_results: Maximum number of detection results
        """
        return vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mode_enum,
            score_threshold=kwargs.get('score_threshold', 0.5),
            max_results=kwargs.get('max_results', 5),
        )

    def _create_detector_instance(self, options: vision.ObjectDetectorOptions) -> vision.ObjectDetector:
        return vision.ObjectDetector.create_from_options(options)

    def _get_options_tuple(self, running_mode: str = None, delegate: str = None, **kwargs) -> tuple:
        if self._current_options:
            return (
                self._current_options.score_threshold,
                self._current_options.max_results,
                running_mode,
                delegate,
            )
        else:
            return (
                kwargs.get('score_threshold', 0.5),
                kwargs.get('max_results', 5),
                running_mode,
                delegate,
            )

    def _process_detection_result(self, detection_result: Any) -> List[ObjectDetectionResult]:
        current_image_detections = []
        if detection_result and detection_result.detections:
            for detection in detection_result.detections:
                bbox_mp = detection.bounding_box
                categories_mp = detection.categories

                bbox = BoundingBox(origin_x=bbox_mp.origin_x, origin_y=bbox_mp.origin_y, width=bbox_mp.width, height=bbox_mp.height)

                categories = [
                    ObjectDetectionCategory(
                        index=cat.index,
                        score=cat.score,
                        display_name=cat.display_name,
                        category_name=cat.category_name,
                    )
                    for cat in categories_mp
                ]

                current_image_detections.append(ObjectDetectionResult(bounding_box=bbox, categories=categories))

        return current_image_detections

    def detect(
        self,
        image: torch.Tensor,
        score_threshold: float = 0.5,
        max_results: int = 5,
        running_mode: str = "video",
        delegate: str = "cpu",
    ) -> List[List[ObjectDetectionResult]]:
        """Detects objects in the input image tensor."""
        return super().detect(
            image,
            running_mode=running_mode,
            delegate=delegate,
            score_threshold=score_threshold,
            max_results=max_results,
        )

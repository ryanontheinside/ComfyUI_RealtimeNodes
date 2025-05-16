from typing import Any, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider
from ..common import BaseDetector
from ..types import LandmarkPoint, PoseLandmarksResult


class PoseLandmarkDetector(BaseDetector[PoseLandmarksResult]):
    """Detects pose landmarks in an image using MediaPipe PoseLandmarker."""

    def __init__(self, model_path: str):
        """Initialize the detector with the model path."""
        super().__init__(model_path)
        self._output_segmentation_masks = False
        self._segmentation_masks = []

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.PoseLandmarkerOptions:
        """Create PoseLandmarker-specific options with parameters:
            - num_poses: Maximum number of poses to detect
            - min_detection_confidence: Minimum confidence for pose detection
            - min_presence_confidence: Minimum confidence for pose presence
            - min_tracking_confidence: Minimum confidence for pose tracking
            - output_segmentation_masks: Whether to output segmentation masks
        """
        self._output_segmentation_masks = kwargs.get('output_segmentation_masks', False)
        return vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            num_poses=kwargs.get('num_poses', 1),
            min_pose_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
            min_pose_presence_confidence=kwargs.get('min_presence_confidence', 0.5),
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            output_segmentation_masks=self._output_segmentation_masks,
        )

    def _create_detector_instance(self, options: vision.PoseLandmarkerOptions) -> vision.PoseLandmarker:
        return vision.PoseLandmarker.create_from_options(options)

    def _get_options_tuple(self, running_mode: str = None, delegate: str = None, **kwargs) -> tuple:
        if self._current_options:
            return (
                self._current_options.num_poses,
                self._current_options.min_pose_detection_confidence,
                self._current_options.min_pose_presence_confidence,
                self._current_options.min_tracking_confidence,
                self._current_options.output_segmentation_masks,
                running_mode,
                delegate,
            )
        else:
            return (
                kwargs.get('num_poses', 1),
                kwargs.get('min_detection_confidence', 0.5),
                kwargs.get('min_presence_confidence', 0.5),
                kwargs.get('min_tracking_confidence', 0.5),
                kwargs.get('output_segmentation_masks', False),
                running_mode,
                delegate,
            )

    def _process_detection_result(self, detection_result: Any) -> List[PoseLandmarksResult]:
        """Process detection result and extract segmentation masks if requested."""
        self._segmentation_masks = []  # Store masks separately
        
        current_image_landmarks = []
        if detection_result and detection_result.pose_landmarks:
            for pose_idx, pose_landmarks_mp in enumerate(detection_result.pose_landmarks):
                landmarks = [
                    LandmarkPoint(
                        index=lm_idx,
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=getattr(lm, "visibility", None),
                        presence=getattr(lm, "presence", None),
                    )
                    for lm_idx, lm in enumerate(pose_landmarks_mp)
                ]

                world_landmarks = None
                if detection_result.pose_world_landmarks and pose_idx < len(detection_result.pose_world_landmarks):
                    world_landmarks = [
                        LandmarkPoint(
                            index=lm_idx,
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=getattr(lm, "visibility", None),
                            presence=getattr(lm, "presence", None),
                        )
                        for lm_idx, lm in enumerate(detection_result.pose_world_landmarks[pose_idx])
                    ]

                current_image_landmarks.append(PoseLandmarksResult(landmarks=landmarks, world_landmarks=world_landmarks))

        # Handle segmentation masks separately
        if self._output_segmentation_masks and detection_result.segmentation_masks:
            mask_mp = detection_result.segmentation_masks[0]  # Get the first mask
            mask_np = mask_mp.numpy_view()
            mask_tensor = torch.from_numpy(mask_np).float()  # HW
            self._segmentation_masks = [mask_tensor]  # Store as a list for consistency

        return current_image_landmarks

    def detect(
        self,
        image: torch.Tensor,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_segmentation_masks: bool = False,
        running_mode: str = "video",
        delegate: str = "cpu",
    ) -> Tuple[List[List[PoseLandmarksResult]], Optional[List[List[torch.Tensor]]]]:
        """Detects pose landmarks in the input image tensor."""
        batch_landmarks = super().detect(
            image,
            running_mode=running_mode,
            delegate=delegate,
            num_poses=num_poses,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )
        
        # Handle segmentation masks
        batch_masks = None
        if output_segmentation_masks:
            batch_size = image.shape[0]
            batch_masks = []
            for i in range(batch_size):
                # We need to build the segmentation masks batch separately
                # since the base detector doesn't know about them
                if hasattr(self, '_segmentation_masks') and self._segmentation_masks:
                    batch_masks.append(self._segmentation_masks)
                else:
                    batch_masks.append([])
                    
            self._segmentation_masks = []  # Clear after use
            
        return batch_landmarks, batch_masks

"""Face Detection implementation using MediaPipe.

This module contains the implementation of MediaPipe Face Detection functionality.
"""

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import List, Tuple
# Import new types
from ..types import FaceDetectionResult, BoundingBox, FaceKeypoint
from ..utils.timestamp_provider import TimestampProvider # Import from utils

class FaceDetector:
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
        self._current_options = None # Store current options for comparison
        self._timestamp_provider = None # Initialize TimestampProvider later

    def _initialize_detector(self, min_detection_confidence: float, running_mode: str, delegate: str):
        """Initializes the FaceDetector."""
        # Map string mode to MediaPipe enum
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == 'cpu' else BaseOptions.Delegate.GPU
        
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mode_enum, # Use enum mode
            min_detection_confidence=min_detection_confidence
        )
        self._current_options = options # Store for comparison
        # Only create timestamp provider for VIDEO mode
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None 
        return vision.FaceDetector.create_from_options(options)

    def detect(self, image: torch.Tensor, min_detection_confidence: float = 0.5, 
                 running_mode: str = "video", delegate: str = 'cpu') -> List[List[FaceDetectionResult]]: # Added running_mode
        """Detects faces in the input image tensor.

        Returns:
            A list (batch) of lists (detections per image) of FaceDetectionResult objects.
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Check if options changed (model path handled implicitly by node)
        # Add running_mode to tuple check
        new_options_tuple = (min_detection_confidence, running_mode, delegate.lower())
        current_options_tuple = None
        if self._current_options:
             current_options_tuple = (
                self._current_options.min_detection_confidence, 
                "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video", # Map enum back to string
                'cpu' if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else 'gpu'
            )

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            # Close previous instance if it exists and has a close method
            if self._detector_instance and hasattr(self._detector_instance, 'close'):
                self._detector_instance.close()
            self._detector_instance = self._initialize_detector(min_detection_confidence, running_mode, delegate)

        batch_size = image.shape[0]
        batch_results = []
        # Determine the correct detection function based on mode
        is_video_mode = self._current_options.running_mode == vision.RunningMode.VIDEO
        detect_func = self._detector_instance.detect_for_video if is_video_mode else self._detector_instance.detect

        for i in range(batch_size):
            img_tensor = image[i] # HWC
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                # Get timestamp for video mode
                timestamp_ms = self._timestamp_provider.next()
                detection_result = detect_func(mp_image, timestamp_ms)
            else:
                # Call image mode function
                detection_result = detect_func(mp_image)
            
            current_image_detections = []
            if detection_result and detection_result.detections: # Check if result is not None
                for detection in detection_result.detections:
                    bbox_mp = detection.bounding_box
                    keypoints_mp = detection.keypoints
                    score = detection.categories[0].score if detection.categories else None # Face detection usually has one score
                    
                    bbox = BoundingBox(
                        origin_x=bbox_mp.origin_x,
                        origin_y=bbox_mp.origin_y,
                        width=bbox_mp.width,
                        height=bbox_mp.height
                    )
                    
                    keypoints = None
                    if keypoints_mp:
                         keypoints = [
                             FaceKeypoint(
                                 label=kp.label,
                                 x=kp.x,
                                 y=kp.y
                             )
                             for kp in keypoints_mp
                         ]

                    current_image_detections.append(FaceDetectionResult(
                        bounding_box=bbox,
                        keypoints=keypoints,
                        score=score
                    ))
            
            batch_results.append(current_image_detections)

        # Check MediaPipe documentation if resource cleanup is required here.
        return batch_results
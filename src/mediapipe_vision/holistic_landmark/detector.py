"""Holistic landmark detector using MediaPipe Legacy Solutions API.

This implementation specifically handles the MediaPipe Holistic solution
which is currently only available through the legacy API, not the Task API.
"""

import mediapipe as mp
import numpy as np
import torch
import cv2
from typing import List, Tuple, Optional, Dict, Any

# Import needed types
from ..types import LandmarkPoint, HolisticLandmarksResult
from ..utils.timestamp_provider import TimestampProvider

class HolisticLandmarkDetector:
    """Detects holistic landmarks combining face, pose and hand tracking.
    
    This detector uses the legacy MediaPipe Solutions API rather than the Task API
    since Holistic is not yet available in the Task API format.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the detector.
        
        Args:
            model_path: Unused parameter to maintain API compatibility with other detectors.
                        The holistic solution doesn't use a model path as it uses the built-in models.
        """
        # Model path is ignored for legacy API - it's for API compatibility only
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = TimestampProvider()

    def _initialize_detector(self, 
                          min_detection_confidence: float, 
                          min_tracking_confidence: float,
                          model_complexity: int,
                          enable_segmentation: bool,
                          refine_face_landmarks: bool,
                          static_image_mode: bool) -> mp.solutions.holistic.Holistic:
        """Initializes the Holistic detector with the appropriate parameters.
        
        Args:
            min_detection_confidence: Minimum confidence for detection to be considered
            min_tracking_confidence: Minimum confidence to maintain tracking
            model_complexity: 0, 1, or 2 determining model complexity
            enable_segmentation: Whether to generate segmentation masks
            refine_face_landmarks: Whether to use refined face landmarks (more details around eyes/lips)
            static_image_mode: If True, treats input as independent images (slower but more accurate)
            
        Returns:
            Initialized Holistic detector instance
        """
        # Store options as a dictionary for parameter comparison
        self._current_options = {
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence,
            'model_complexity': model_complexity,
            'enable_segmentation': enable_segmentation,
            'refine_face_landmarks': refine_face_landmarks,
            'static_image_mode': static_image_mode
        }
        
        # Create the holistic solution instance with specified options
        return mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image: torch.Tensor, 
              min_detection_confidence: float = 0.5,
              min_tracking_confidence: float = 0.5,
              model_complexity: int = 1,
              enable_segmentation: bool = False,
              refine_face_landmarks: bool = False,
              static_image_mode: bool = False) -> Tuple[List[HolisticLandmarksResult], Optional[List[torch.Tensor]]]:
        """Detects holistic landmarks in the input image tensor.
        
        Args:
            image: Input image tensor in BHWC format
            min_detection_confidence: Minimum confidence for detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0)
            model_complexity: Complexity of the pose landmark model (0, 1, or 2)
            enable_segmentation: Whether to output segmentation masks
            refine_face_landmarks: Whether to use refined face landmarks
            static_image_mode: If True, treats input as independent images
            
        Returns:
            Tuple containing:
                - holistic_landmarks_batch: List of HolisticLandmarksResult objects for each image
                - segmentation_masks_batch: List of segmentation masks (if requested)
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Create a parameters tuple for comparison
        new_options = {
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence,
            'model_complexity': model_complexity,
            'enable_segmentation': enable_segmentation,
            'refine_face_landmarks': refine_face_landmarks,
            'static_image_mode': static_image_mode
        }

        # Initialize or update detector if needed
        if self._detector_instance is None or self._current_options != new_options:
            if self._detector_instance and hasattr(self._detector_instance, 'close'):
                self._detector_instance.close()
            self._detector_instance = self._initialize_detector(
                min_detection_confidence,
                min_tracking_confidence,
                model_complexity,
                enable_segmentation,
                refine_face_landmarks,
                static_image_mode
            )

        batch_size = image.shape[0]
        batch_holistic_landmarks = []
        batch_segmentation_masks = [] if enable_segmentation else None
        
        for i in range(batch_size):
            img_tensor = image[i]  # HWC
            # Convert to numpy with proper scaling and datatype
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Process the image with MediaPipe Holistic
            holistic_results = self._detector_instance.process(
                cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            )
            
            # Convert results to our standardized format
            holistic_result = self._convert_to_landmark_result(holistic_results)
            batch_holistic_landmarks.append(holistic_result)
            
            # Handle segmentation mask if requested
            if enable_segmentation and holistic_results.segmentation_mask is not None:
                mask_np = holistic_results.segmentation_mask
                mask_tensor = torch.from_numpy(mask_np).float()  # HW
                if batch_segmentation_masks is not None:
                    batch_segmentation_masks.append(mask_tensor)
        
        return batch_holistic_landmarks, batch_segmentation_masks

    def _convert_to_landmark_result(self, holistic_results: Any) -> HolisticLandmarksResult:
        """Converts MediaPipe Holistic results to standardized HolisticLandmarksResult.
        
        Args:
            holistic_results: Raw results from mp.solutions.holistic
            
        Returns:
            Standardized HolisticLandmarksResult containing all landmarks
        """
        # Convert face landmarks
        face_landmarks = None
        if holistic_results.face_landmarks:
            face_landmarks = [
                LandmarkPoint(
                    index=idx,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=getattr(landmark, 'visibility', None)
                )
                for idx, landmark in enumerate(holistic_results.face_landmarks.landmark)
            ]
        
        # Convert pose landmarks
        pose_landmarks = None
        if holistic_results.pose_landmarks:
            pose_landmarks = [
                LandmarkPoint(
                    index=idx,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility
                )
                for idx, landmark in enumerate(holistic_results.pose_landmarks.landmark)
            ]
        
        # Convert pose world landmarks
        pose_world_landmarks = None
        if holistic_results.pose_world_landmarks:
            pose_world_landmarks = [
                LandmarkPoint(
                    index=idx,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility
                )
                for idx, landmark in enumerate(holistic_results.pose_world_landmarks.landmark)
            ]
        
        # Convert left hand landmarks
        left_hand_landmarks = None
        if holistic_results.left_hand_landmarks:
            left_hand_landmarks = [
                LandmarkPoint(
                    index=idx,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                )
                for idx, landmark in enumerate(holistic_results.left_hand_landmarks.landmark)
            ]
        
        # Convert right hand landmarks
        right_hand_landmarks = None
        if holistic_results.right_hand_landmarks:
            right_hand_landmarks = [
                LandmarkPoint(
                    index=idx,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                )
                for idx, landmark in enumerate(holistic_results.right_hand_landmarks.landmark)
            ]
        
        return HolisticLandmarksResult(
            face_landmarks=face_landmarks,
            pose_landmarks=pose_landmarks,
            pose_world_landmarks=pose_world_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks
        )
    
    def close(self):
        """Releases resources used by the detector."""
        if self._detector_instance and hasattr(self._detector_instance, 'close'):
            self._detector_instance.close()
            self._detector_instance = None 
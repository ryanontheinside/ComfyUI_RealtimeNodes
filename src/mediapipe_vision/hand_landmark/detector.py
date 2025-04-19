import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import List, Tuple, Optional
# Import new types
from ..types import LandmarkPoint, HandLandmarksResult
from ..utils.timestamp_provider import TimestampProvider # Import TimestampProvider

class HandLandmarkDetector:
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
        self._timestamp_provider = None # Added

    def _initialize_detector(self, num_hands: int, min_detection_confidence: float, 
                             min_presence_confidence: float, min_tracking_confidence: float, 
                             running_mode: str, delegate: str): # Added running_mode
        """Initializes the HandLandmarker detector."""
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO # Added
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == 'cpu' else BaseOptions.Delegate.GPU
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum, # Use enum mode
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            # min_hand_presence_confidence and min_tracking_confidence are for VIDEO/LIVE_STREAM
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence      
        )
        self._current_options = options
        # Only create timestamp provider for VIDEO mode
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None 
        return vision.HandLandmarker.create_from_options(options)

    def detect(self, image: torch.Tensor, num_hands: int = 2, min_detection_confidence: float = 0.5, 
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5, 
                 running_mode: str = "video", delegate: str = 'cpu') -> List[List[HandLandmarksResult]]: # Added running_mode
        """Detects hand landmarks in the input image tensor.

        Returns:
            A list (batch) of lists (hands per image) of HandLandmarksResult objects.
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Add running_mode to tuple check
        new_options_tuple = (num_hands, min_detection_confidence, 
                             min_presence_confidence, min_tracking_confidence, running_mode, delegate.lower())
        current_options_tuple = None
        if self._current_options:
             current_options_tuple = (
                 self._current_options.num_hands,
                 self._current_options.min_hand_detection_confidence,
                 self._current_options.min_hand_presence_confidence, 
                 self._current_options.min_tracking_confidence,    
                 "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video", # Map enum back to string
                 'cpu' if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else 'gpu'
             )

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
             if self._detector_instance and hasattr(self._detector_instance, 'close'):
                  self._detector_instance.close()
             # Pass new params to initializer
             self._detector_instance = self._initialize_detector(num_hands, min_detection_confidence, 
                                                                 min_presence_confidence, min_tracking_confidence, 
                                                                 running_mode, delegate) # Pass running_mode

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
                timestamp_ms = self._timestamp_provider.next()
                detection_result = detect_func(mp_image, timestamp_ms)
            else:
                detection_result = detect_func(mp_image)
            
            current_image_results = []
            if detection_result and detection_result.hand_landmarks: # Added check for detection_result
                for hand_idx, hand_landmarks_mp in enumerate(detection_result.hand_landmarks):
                    
                    landmarks = [
                        LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z)
                        for lm_idx, lm in enumerate(hand_landmarks_mp)
                    ]
                    
                    world_landmarks = None
                    if detection_result.hand_world_landmarks and hand_idx < len(detection_result.hand_world_landmarks):
                        world_landmarks = [
                            LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z)
                            for lm_idx, lm in enumerate(detection_result.hand_world_landmarks[hand_idx])
                        ]
                    
                    handedness = None
                    if detection_result.handedness and hand_idx < len(detection_result.handedness):
                        handedness_cats = detection_result.handedness[hand_idx]
                        if handedness_cats: handedness = handedness_cats[0].display_name
                    
                    current_image_results.append(HandLandmarksResult(
                        landmarks=landmarks,
                        world_landmarks=world_landmarks,
                        handedness=handedness
                    ))
            
            batch_results.append(current_image_results)

        # self._detector_instance.close() # Close handled on re-init
        return batch_results 
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import List
# Import new types
from ..types import GestureRecognitionResult, GestureCategory
from ..utils.timestamp_provider import TimestampProvider # Import TimestampProvider

class GestureRecognizer:
    """Recognizes hand gestures using MediaPipe GestureRecognizer."""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = None # Added

    def _initialize_detector(self, num_hands: int, min_detection_confidence: float, 
                             min_tracking_confidence: float, min_presence_confidence: float, 
                             running_mode: str, delegate: str): # Added running_mode
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO # Added
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == 'cpu' else BaseOptions.Delegate.GPU
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        # Note: GestureRecognizerOptions has min_tracking_confidence, useful for VIDEO mode
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mode_enum, # Use enum mode
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence, 
            min_hand_presence_confidence=min_presence_confidence 
            # Add other relevant VIDEO mode parameters if needed (e.g., canned_gestures_classifier_options)
        )
        self._current_options = options
        # Only create timestamp provider for VIDEO mode
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None 
        return vision.GestureRecognizer.create_from_options(options)

    def recognize(self, image: torch.Tensor, num_hands: int = 2, min_detection_confidence: float = 0.5, 
                  min_tracking_confidence: float = 0.5, min_presence_confidence: float = 0.5, 
                  running_mode: str = "video", delegate: str = 'cpu') -> List[List[GestureRecognitionResult]]: # Added running_mode
        """Recognizes hand gestures in the input image tensor."""
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")
            
        # Add running_mode to tuple check
        new_options_tuple = (num_hands, min_detection_confidence, min_tracking_confidence, 
                             min_presence_confidence, running_mode, delegate.lower())
        current_options_tuple = None
        if self._current_options:
             current_options_tuple = (
                 self._current_options.num_hands,
                 self._current_options.min_hand_detection_confidence,
                 self._current_options.min_tracking_confidence,
                 self._current_options.min_hand_presence_confidence,
                 "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video", # Map enum back to string
                 'cpu' if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else 'gpu'
             )

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            if self._detector_instance and hasattr(self._detector_instance, 'close'):
                 self._detector_instance.close()
            # Pass new params to initializer
            self._detector_instance = self._initialize_detector(num_hands, min_detection_confidence, 
                                                              min_tracking_confidence, min_presence_confidence, 
                                                              running_mode, delegate) # Pass running_mode

        batch_size = image.shape[0]
        batch_results = []
        # Determine the correct recognition function based on mode
        is_video_mode = self._current_options.running_mode == vision.RunningMode.VIDEO
        recognize_func = self._detector_instance.recognize_for_video if is_video_mode else self._detector_instance.recognize

        for i in range(batch_size):
            img_tensor = image[i] 
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                timestamp_ms = self._timestamp_provider.next()
                recognition_result = recognize_func(mp_image, timestamp_ms)
            else:
                recognition_result = recognize_func(mp_image)
            
            current_image_gestures = []
            if recognition_result and recognition_result.gestures: # Added check for recognition_result
                for hand_idx, hand_gestures in enumerate(recognition_result.gestures):
                    gesture_categories = [GestureCategory(index=cat.index, score=cat.score, display_name=cat.display_name, category_name=cat.category_name) for cat in hand_gestures]
                    handedness = None
                    if recognition_result.handedness and hand_idx < len(recognition_result.handedness):
                        handedness_cats = recognition_result.handedness[hand_idx]
                        if handedness_cats: handedness = handedness_cats[0].display_name
                    current_image_gestures.append(GestureRecognitionResult(gestures=gesture_categories, handedness=handedness))
            batch_results.append(current_image_gestures)

        # self._detector_instance.close() # Close handled on re-init
        return batch_results 
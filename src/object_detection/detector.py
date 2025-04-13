import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import List
# Import new types
from ..types import ObjectDetectionResult, ObjectDetectionCategory, BoundingBox
from ..utils.timestamp_provider import TimestampProvider # Import TimestampProvider

class ObjectDetector:
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
        self._timestamp_provider = None # Added

    def _initialize_detector(self, score_threshold: float, max_results: int, 
                             running_mode: str, delegate: str): # Added running_mode
        """Initializes the ObjectDetector."""
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO # Added
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == 'cpu' else BaseOptions.Delegate.GPU
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mode_enum, # Use enum mode
            score_threshold=score_threshold,
            max_results=max_results
        )
        self._current_options = options
        # Only create timestamp provider for VIDEO mode
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None 
        return vision.ObjectDetector.create_from_options(options)

    def detect(self, image: torch.Tensor, score_threshold: float = 0.5, max_results: int = 5, 
                 running_mode: str = "video", delegate: str = 'cpu') -> List[List[ObjectDetectionResult]]: # Added running_mode
        """Detects objects in the input image tensor.

        Returns:
            A list (batch) of lists (detections per image) of ObjectDetectionResult objects.
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Add running_mode to tuple check
        new_options_tuple = (score_threshold, max_results, running_mode, delegate.lower())
        current_options_tuple = None
        if self._current_options:
             current_options_tuple = (
                 self._current_options.score_threshold,
                 self._current_options.max_results,
                 "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video", # Map enum back to string
                 'cpu' if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else 'gpu'
             )

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            if self._detector_instance and hasattr(self._detector_instance, 'close'):
                 self._detector_instance.close()
            self._detector_instance = self._initialize_detector(score_threshold, max_results, running_mode, delegate)

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
            
            current_image_detections = []
            if detection_result and detection_result.detections: # Added check for detection_result
                for detection in detection_result.detections:
                    bbox_mp = detection.bounding_box
                    categories_mp = detection.categories
                    
                    bbox = BoundingBox(
                        origin_x=bbox_mp.origin_x,
                        origin_y=bbox_mp.origin_y,
                        width=bbox_mp.width,
                        height=bbox_mp.height
                    )
                    
                    categories = [
                        ObjectDetectionCategory(
                            index=cat.index, 
                            score=cat.score,
                            display_name=cat.display_name, 
                            category_name=cat.category_name
                        )
                        for cat in categories_mp
                    ]
                    
                    current_image_detections.append(ObjectDetectionResult(
                        bounding_box=bbox,
                        categories=categories
                    ))
            
            batch_results.append(current_image_detections)

        # self._detector_instance.close() # Close handled on re-init
        return batch_results 
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider

T = TypeVar('T')  # Generic type for detector results

class BaseDetector(Generic[T], ABC):
    """Base class for MediaPipe vision detectors.
    
    This class provides common functionality for all MediaPipe vision detectors,
    reducing code duplication and providing a consistent interface.
    
    Generic type T represents the detector-specific result type.
    """

    def __init__(self, model_path: str):
        """Initialize the detector with the model path.
        
        Args:
            model_path: Path to the MediaPipe model file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        
        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = None

    @abstractmethod
    def _create_detector_options(self, base_options: python.BaseOptions, 
                                mode_enum: vision.RunningMode, **kwargs) -> Any:
        """Create detector-specific options.
        
        Args:
            base_options: MediaPipe base options with model path and delegate.
            mode_enum: RunningMode enum (IMAGE or VIDEO).
            **kwargs: Detector-specific configuration parameters.
            
        Returns:
            Detector-specific options object.
        """
        pass

    @abstractmethod
    def _get_options_tuple(self, running_mode: str = None, 
                          delegate: str = None, **kwargs) -> tuple:
        """Get tuple of options for comparison to detect config changes.
        
        Args:
            running_mode: String representation of running mode ("image" or "video").
            delegate: String representation of delegate ("cpu" or "gpu").
            **kwargs: Detector-specific parameters to include in comparison.
            
        Returns:
            Tuple containing all configuration options for comparison.
        """
        pass

    @abstractmethod
    def _process_detection_result(self, detection_result: Any) -> List[T]:
        """Process detector-specific results.
        
        Args:
            detection_result: Raw detection result from MediaPipe.
            
        Returns:
            List of processed detection results of type T.
        """
        pass

    @abstractmethod
    def _create_detector_instance(self, options: Any) -> Any:
        """Create detector-specific instance.
        
        Args:
            options: Configured detector options.
            
        Returns:
            MediaPipe detector instance.
        """
        pass

    def _initialize_detector(self, running_mode: str, delegate: str, **kwargs) -> Any:
        """Initialize the MediaPipe detector with specified options.
        
        Args:
            running_mode: "image" or "video".
            delegate: "cpu" or "gpu".
            **kwargs: Detector-specific configuration parameters.
            
        Returns:
            Initialized detector instance.
        """
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == "cpu" else BaseOptions.Delegate.GPU
        
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = self._create_detector_options(base_options, mode_enum, **kwargs)
        
        self._current_options = options
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None
        
        return self._create_detector_instance(options)

    def detect(self, image: torch.Tensor, running_mode: str = "video", 
              delegate: str = "cpu", **kwargs) -> List[List[T]]:
        """Detect features in the input image tensor.
        
        Args:
            image: Input image tensor in BHWC format.
            running_mode: "image" or "video".
            delegate: "cpu" or "gpu".
            **kwargs: Detector-specific parameters.
            
        Returns:
            A list (batch) of lists (detections per image) of detection results.
            
        Raises:
            ValueError: If the input tensor is not in the expected format.
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Check if options changed to decide whether to reinitialize
        new_options_tuple = self._get_options_tuple(running_mode=running_mode, delegate=delegate, **kwargs)
        current_options_tuple = None
        if self._current_options:
            current_mode = "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video"
            current_delegate = "cpu" if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else "gpu"
            current_options_tuple = self._get_options_tuple(running_mode=current_mode, delegate=current_delegate)

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            if self._detector_instance and hasattr(self._detector_instance, "close"):
                self._detector_instance.close()
            self._detector_instance = self._initialize_detector(running_mode, delegate, **kwargs)

        batch_size = image.shape[0]
        batch_results = []
        is_video_mode = self._current_options.running_mode == vision.RunningMode.VIDEO
        
        # Get the appropriate detection function based on mode
        if hasattr(self._detector_instance, "detect_for_video") and is_video_mode:
            detect_func = self._detector_instance.detect_for_video
        elif hasattr(self._detector_instance, "detect"):
            detect_func = self._detector_instance.detect
        else:
            # Try to find the appropriate method name for this detector type
            func_names = [func for func in dir(self._detector_instance) if func.startswith(("detect", "recognize")) and callable(getattr(self._detector_instance, func))]
            video_funcs = [func for func in func_names if "video" in func.lower()]
            detect_func = getattr(self._detector_instance, video_funcs[0] if is_video_mode and video_funcs else func_names[0])

        # Process each image in the batch
        for i in range(batch_size):
            img_tensor = image[i]
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                timestamp_ms = self._timestamp_provider.next()
                detection_result = detect_func(mp_image, timestamp_ms)
            else:
                detection_result = detect_func(mp_image)

            batch_results.append(self._process_detection_result(detection_result))

        return batch_results 
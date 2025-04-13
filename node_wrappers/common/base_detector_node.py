"""Base node wrapper for MediaPipe Vision detectors."""

import torch
import logging
from typing import Any, Dict, List, Tuple, Optional, Type, Callable, Union

logger = logging.getLogger(__name__)

class BaseMediaPipeDetectorNode:
    """Base class for all MediaPipe detector nodes in ComfyUI.
    
    This class captures common patterns across detector nodes like:
    - Detector instance management
    - Model validation
    - Common input parameters
    - Cleanup in __del__
    
    Parameter Naming Standards for Subclasses:
    -----------------------------------------
    To maintain consistency across detector types, use these standardized parameter names:
    
    - min_confidence: Main confidence threshold for detection (replaces various min_detection_confidence, 
                     score_threshold, etc.)
    - min_tracking_confidence: Confidence threshold for tracking objects across video frames
    - min_presence_confidence: Confidence threshold for presence of an object in a frame
    - max_results: Maximum number of detections to return (replaces num_poses, num_hands, etc.)
    
    When implementing specific detector nodes, map these standardized parameter names to 
    the appropriate underlying MediaPipe API parameters in the detect() method.
    """
    
    # To be overridden by subclasses
    DETECTOR_CLASS = None  # Subclasses must set to their specific detector class
    MODEL_INFO_TYPE = "MEDIAPIPE_MODEL_INFO"  # Override with specific model info type
    EXPECTED_TASK_TYPE = None  # Must be set to match the MODEL_INFO_TYPE task_type field
    CATEGORY = "MediaPipeVision/Base"  # Override with specific category
    RETURN_TYPES = tuple()  # Must be set by subclass
    RETURN_NAMES = tuple()  # Must be set by subclass
    FUNCTION = "detect"  # Default function name, can be overridden
    DESCRIPTION = ""  # Override with specific node description
    
    def __init__(self):
        """Initialize with empty detector and model path."""
        self._detector = None
        self._model_path = None
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define common input types for detector nodes."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for detection"}),
                "model_info": (cls.MODEL_INFO_TYPE, {"tooltip": f"Model loaded from the appropriate loader node"}),
                "running_mode": (["image", "video"], {
                    "default": "video", 
                    "tooltip": "Processing mode: 'image' for single images, 'video' for sequences/streams"
                }),
                "delegate": (["cpu", "gpu"], {
                    "default": "cpu", 
                    "tooltip": "Processing device - GPU is faster but CPU is more compatible with all systems"
                }),
            }
        }
        # Subclasses should extend this with additional parameters
    
    def validate_model_info(self, model_info: Dict) -> str:
        """Validate model_info and return model path."""
        if not isinstance(model_info, dict):
            raise ValueError(f"Invalid model_info provided. Expected dictionary, got {type(model_info)}")
            
        task_type = model_info.get('task_type')
        if task_type != self.EXPECTED_TASK_TYPE:
            raise ValueError(f"Invalid model_info provided. Expected task_type '{self.EXPECTED_TASK_TYPE}', got '{task_type}'")
        
        model_path = model_info.get('model_path')
        if not model_path:
            raise ValueError("Model path not found or invalid in model_info")
            
        return model_path
    
    def initialize_or_update_detector(self, model_path: str):
        """Initialize detector or update if model path changed."""
        if self.DETECTOR_CLASS is None:
            raise NotImplementedError("Subclasses must set DETECTOR_CLASS to their specific detector class")
            
        if self._detector is None or self._model_path != model_path:
            # Close existing detector if needed
            if self._detector and hasattr(self._detector, 'close'):
                try:
                    logger.info(f"Closing existing detector for {self._model_path}")
                    self._detector.close()
                except Exception as e:
                    logger.warning(f"Error closing detector: {e}")
            
            # Create new detector
            logger.info(f"Creating new {self.DETECTOR_CLASS.__name__} instance for {model_path}")
            self._detector = self.DETECTOR_CLASS(model_path)
            self._model_path = model_path
        elif not hasattr(self._detector, '_detector_instance'):
            # Re-initialize if internal instance is missing
            logger.warning(f"Re-initializing detector wrapper due to missing internal instance")
            self._detector = self.DETECTOR_CLASS(model_path)
            self._model_path = model_path
        
        return self._detector
    
    def detect(self, **kwargs):
        """
        Main detection function, should be implemented by subclasses.
        This is a placeholder that subclasses can use as a template.
        """
        # Basic implementation that subclasses can override
        
        # 1. Extract and validate model_info
        model_info = kwargs.get('model_info')
        model_path = self.validate_model_info(model_info)
        
        # 2. Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # 3. Call detector's detect method with appropriate parameters
        # This part should be implemented by subclasses
        
        raise NotImplementedError("Subclasses must implement or override the detect method")
    
    def __del__(self):
        """Clean up resources by closing detector."""
        if hasattr(self, '_detector') and self._detector and hasattr(self._detector, 'close'):
            try:
                logger.info(f"Closing detector instance in __del__ for {self._model_path}")
                self._detector.close()
            except Exception as e:
                logger.warning(f"Error closing detector in __del__: {e}")
            self._detector = None 
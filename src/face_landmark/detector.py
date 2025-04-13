import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import List, Tuple, Optional, Dict
# Import new types
from ..types import LandmarkPoint, FaceLandmarksResult
from ..utils.timestamp_provider import TimestampProvider # Import TimestampProvider

class FaceLandmarkDetector:
    """Detects face landmarks and blendshapes using MediaPipe FaceLandmarker."""

    def __init__(self, model_path: str):
        """Initialize the detector with the model path.

        Args:
            model_path: Path to the MediaPipe FaceLandmarker .task file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")

        self.model_path = model_path
        self._detector_instance = None 
        self._current_options = None # Store current options for comparison
        self._timestamp_provider = None # Added

    def _initialize_detector(self, num_faces: int, min_detection_confidence: float, 
                             min_presence_confidence: float, min_tracking_confidence: float, 
                             output_blendshapes: bool, output_transform_matrix: bool, 
                             running_mode: str, delegate: str): # Added running_mode
        """Initializes the FaceLandmarker detector."""
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO # Added
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == 'cpu' else BaseOptions.Delegate.GPU
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum, # Use enum mode
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            # min_face_presence_confidence and min_tracking_confidence are for VIDEO/LIVE_STREAM
            min_face_presence_confidence=min_presence_confidence, 
            min_tracking_confidence=min_tracking_confidence,      
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=output_transform_matrix
        )
        self._current_options = options # Store for comparison
        # Only create timestamp provider for VIDEO mode
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None 
        return vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: torch.Tensor, num_faces: int = 1, min_detection_confidence: float = 0.5, 
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5, 
                 output_blendshapes: bool = False, output_transform_matrix: bool = False,
                 running_mode: str = "video", delegate: str = 'cpu') -> Tuple[List[List[LandmarkPoint]], Optional[List[Dict[str, float]]], Optional[List[np.ndarray]]]: # Added running_mode
        """Detects face landmarks in the input image tensor.

        Returns:
            A tuple containing:
                - face_landmarks_batch: List (batch) of lists (faces) of LandmarkPoint objects.
                - blendshapes: List of dictionaries mapping blendshape names to scores (if requested).
                - transform_matrices: List of transformation matrices (NumPy arrays) (if requested).
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")
            
        # Check if options changed 
        # Add running_mode to tuple check
        new_options_tuple = (num_faces, min_detection_confidence, min_presence_confidence, min_tracking_confidence, 
                             output_blendshapes, output_transform_matrix, running_mode, delegate.lower())
        current_options_tuple = None
        if self._current_options:
            current_options_tuple = (
                self._current_options.num_faces,
                self._current_options.min_face_detection_confidence,
                self._current_options.min_face_presence_confidence, 
                self._current_options.min_tracking_confidence,    
                self._current_options.output_face_blendshapes,
                self._current_options.output_facial_transformation_matrixes,
                "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video", # Map enum back to string
                'cpu' if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else 'gpu'
            )

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            if self._detector_instance and hasattr(self._detector_instance, 'close'): # Close previous instance if options changed
                 self._detector_instance.close()
            # Pass new params to initializer
            self._detector_instance = self._initialize_detector(num_faces, min_detection_confidence, 
                                                              min_presence_confidence, min_tracking_confidence, 
                                                              output_blendshapes, output_transform_matrix, 
                                                              running_mode, delegate) # Pass running_mode

        batch_size = image.shape[0]
        batch_results_landmarks = []
        batch_results_blendshapes = [] if output_blendshapes else None
        batch_results_matrices = [] if output_transform_matrix else None

        # Determine the correct detection function based on mode
        is_video_mode = self._current_options.running_mode == vision.RunningMode.VIDEO
        detect_func = self._detector_instance.detect_for_video if is_video_mode else self._detector_instance.detect
        
        for i in range(batch_size):
            img_tensor = image[i] 
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                timestamp_ms = self._timestamp_provider.next()
                detection_result = detect_func(mp_image, timestamp_ms)
            else:
                detection_result = detect_func(mp_image)
            
            current_image_landmarks = []
            current_image_blendshapes = [] if output_blendshapes else None
            current_image_matrices = [] if output_transform_matrix else None
            if detection_result and detection_result.face_landmarks: # Added check for detection_result
                for face_idx, face_landmarks_mp in enumerate(detection_result.face_landmarks):
                    landmarks = [LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z) for lm_idx, lm in enumerate(face_landmarks_mp)]
                    current_image_landmarks.append(landmarks)
                    if output_blendshapes and detection_result.face_blendshapes and face_idx < len(detection_result.face_blendshapes):
                         blendshapes = {bs.category_name: bs.score for bs in detection_result.face_blendshapes[face_idx]}
                         current_image_blendshapes.append(blendshapes)
                    if output_transform_matrix and detection_result.facial_transformation_matrixes and face_idx < len(detection_result.facial_transformation_matrixes):
                         current_image_matrices.append(detection_result.facial_transformation_matrixes[face_idx])
            batch_results_landmarks.append(current_image_landmarks)
            if output_blendshapes: batch_results_blendshapes.append(current_image_blendshapes)
            if output_transform_matrix: batch_results_matrices.append(current_image_matrices)
                 
        # Close detector after batch? Landmarker might need closing per batch? Check docs.
        # For IMAGE mode, closing after batch seems okay.
        # self._detector_instance.close() # Close is handled if re-initialized
        return batch_results_landmarks, batch_results_blendshapes, batch_results_matrices 
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import List, Tuple, Optional
# Import new types
from ..types import LandmarkPoint, PoseLandmarksResult
from ..utils.timestamp_provider import TimestampProvider # Import TimestampProvider

class PoseLandmarkDetector:
    """Detects pose landmarks in an image using MediaPipe PoseLandmarker."""
    
    def __init__(self, model_path: str):
        """Initialize the detector with the model path.

        Args:
            model_path: Path to the MediaPipe PoseLandmarker .task file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        
        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = None # Added

    def _initialize_detector(self, num_poses: int, min_detection_confidence: float, 
                             min_presence_confidence: float, min_tracking_confidence: float, 
                             output_segmentation_masks: bool, 
                             running_mode: str, delegate: str) -> vision.PoseLandmarker: # Added running_mode
        """Initializes the PoseLandmarker detector."""
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO # Added
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == 'cpu' else BaseOptions.Delegate.GPU
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum, # Use enum mode
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence, 
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks
        )
        self._current_options = options
        # Only create timestamp provider for VIDEO mode
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None 
        return vision.PoseLandmarker.create_from_options(options)

    def detect(self, image: torch.Tensor, num_poses: int = 1, min_detection_confidence: float = 0.5, 
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5, 
                 output_segmentation_masks: bool = False, 
                 running_mode: str = "video", delegate: str = 'cpu') -> Tuple[List[List[PoseLandmarksResult]], Optional[List[List[torch.Tensor]]]]: # Added running_mode
        """Detects pose landmarks in the input image tensor.

        Returns:
            A tuple containing:
                - pose_landmarks_batch: List (batch) of lists (poses) of PoseLandmarksResult objects.
                - segmentation_masks_batch: List (batch) of lists (poses) of segmentation masks (HW FloatTensor) (if requested).
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Add running_mode to tuple check
        new_options_tuple = (num_poses, min_detection_confidence, min_presence_confidence, 
                             min_tracking_confidence, output_segmentation_masks, running_mode, delegate.lower())
        current_options_tuple = None
        if self._current_options:
             current_options_tuple = (
                 self._current_options.num_poses,
                 self._current_options.min_pose_detection_confidence,
                 self._current_options.min_pose_presence_confidence,
                 self._current_options.min_tracking_confidence, 
                 self._current_options.output_segmentation_masks,
                 "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video", # Map enum back to string
                 'cpu' if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else 'gpu'
             )

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            if self._detector_instance and hasattr(self._detector_instance, 'close'):
                 self._detector_instance.close()
            # Pass new params to initializer
            self._detector_instance = self._initialize_detector(num_poses, min_detection_confidence, 
                                                              min_presence_confidence, min_tracking_confidence, 
                                                              output_segmentation_masks, running_mode, delegate) # Pass running_mode

        batch_size = image.shape[0]
        batch_results_landmarks = []
        batch_results_masks = [] if output_segmentation_masks else None
        
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

            current_image_landmarks = []
            current_image_masks = [] if output_segmentation_masks else None

            if detection_result and detection_result.pose_landmarks: # Added check for detection_result
                for pose_idx, pose_landmarks_mp in enumerate(detection_result.pose_landmarks):
                    landmarks = [
                        LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z, 
                                      visibility=getattr(lm, 'visibility', None), # Use getattr for safety
                                      presence=getattr(lm, 'presence', None))
                        for lm_idx, lm in enumerate(pose_landmarks_mp)
                    ]
                    
                    world_landmarks = None
                    if detection_result.pose_world_landmarks and pose_idx < len(detection_result.pose_world_landmarks):
                         world_landmarks = [
                             LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z,
                                           visibility=getattr(lm, 'visibility', None),
                                           presence=getattr(lm, 'presence', None))
                             for lm_idx, lm in enumerate(detection_result.pose_world_landmarks[pose_idx])
                         ]
                    
                    current_image_landmarks.append(PoseLandmarksResult(
                        landmarks=landmarks,
                        world_landmarks=world_landmarks
                    ))

            # Handle segmentation masks separately as they are not indexed per pose in the result
            if output_segmentation_masks and detection_result.segmentation_masks:
                 # MediaPipe returns one mask per image if output_segmentation_masks is True
                 # We store it as a list associated with the image, not per pose
                 # Assuming only one mask is relevant per image, might need adjustment if multiple masks are returned
                 mask_mp = detection_result.segmentation_masks[0] # Get the first mask
                 mask_np = mask_mp.numpy_view()
                 mask_tensor = torch.from_numpy(mask_np).float() # HW
                 current_image_masks = [mask_tensor] # Store as a list for consistency with return type
            
            batch_results_landmarks.append(current_image_landmarks)
            if output_segmentation_masks:
                batch_results_masks.append(current_image_masks)

        # self._detector_instance.close() # Close handled on re-init
        return batch_results_landmarks, batch_results_masks 
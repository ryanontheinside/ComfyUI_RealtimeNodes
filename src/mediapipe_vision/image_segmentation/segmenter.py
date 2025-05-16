"""Image Segmentation implementation using MediaPipe.

This module contains the implementation of MediaPipe Image Segmentation functionality.
"""

import logging
import platform  # Needed for OS check
from typing import List, Tuple

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ImageSegmenter as MPImageSegmenter

from ..common.base_detector import BaseDetector
from ...utils.timing import TimestampProvider

logger = logging.getLogger(__name__)


class ImageSegmenter(BaseDetector):
    """Performs image segmentation using MediaPipe ImageSegmenter.
    Adapted from ComfyUI-Stream-Pack implementation.
    """

    def _create_detector_options(self, base_options: python.BaseOptions, 
                                mode_enum: vision.RunningMode, **kwargs) -> vision.ImageSegmenterOptions:
        """Create image segmenter-specific options.
        
        Args:
            base_options: MediaPipe base options with model path and delegate.
            mode_enum: RunningMode enum (IMAGE or VIDEO).
            **kwargs: Detector-specific configuration parameters.
            
        Returns:
            Configured ImageSegmenterOptions object.
        """
        output_confidence_masks = kwargs.get('output_confidence_masks', True)
        output_category_mask = kwargs.get('output_category_mask', False)
        
        return vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mode_enum,
            output_confidence_masks=output_confidence_masks,
            output_category_mask=output_category_mask,
        )

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
        output_confidence_masks = kwargs.get('output_confidence_masks', True)
        output_category_mask = kwargs.get('output_category_mask', False)
        
        # Windows GPU check - provide warning and fallback
        if delegate and delegate.lower() == "gpu" and platform.system().lower() == "windows":
            logger.warning("GPU delegate is not supported on Windows for Python MediaPipe Tasks. Falling back to CPU.")
            delegate = "cpu"
        
        return (output_confidence_masks, output_category_mask, running_mode, delegate)

    def _process_detection_result(self, detection_result) -> tuple:
        """Process segmentation results.
        
        Args:
            detection_result: Raw segmentation result from MediaPipe.
            
        Returns:
            Tuple of (confidence_masks, category_mask) as torch tensors or None.
        """
        confidence_masks = None
        category_mask = None
        
        if self._current_options and hasattr(self._current_options, 'output_confidence_masks') and self._current_options.output_confidence_masks:
            if detection_result and detection_result.confidence_masks:
                confidence_masks = []
                for mask in detection_result.confidence_masks:
                    mask_np = mask.numpy_view()
                    mask_tensor = torch.from_numpy(mask_np).float()  # Keep as HW
                    confidence_masks.append(mask_tensor)
        
        if self._current_options and hasattr(self._current_options, 'output_category_mask') and self._current_options.output_category_mask:
            if detection_result and detection_result.category_mask:
                mask_np = detection_result.category_mask.numpy_view()
                category_mask = torch.from_numpy(mask_np).long()  # Keep as HW
        
        return confidence_masks, category_mask

    def _create_detector_instance(self, options: vision.ImageSegmenterOptions) -> vision.ImageSegmenter:
        """Create segmenter instance.
        
        Args:
            options: Configured ImageSegmenterOptions.
            
        Returns:
            MediaPipe ImageSegmenter instance.
        """
        try:
            return vision.ImageSegmenter.create_from_options(options)
        except Exception as e:
            logger.error(f"Failed to create ImageSegmenter: {e}")
            raise RuntimeError(f"Failed to initialize MediaPipe ImageSegmenter: {e}")

    def detect(self, image: torch.Tensor, running_mode: str = "video", 
              delegate: str = "cpu", **kwargs) -> List[tuple]:
        """Override of the BaseDetector detect method to use segment/segment_for_video.
        
        Args:
            image: Input image tensor in BHWC format.
            running_mode: "image" or "video".
            delegate: "cpu" or "gpu".
            **kwargs: Detector-specific parameters.
            
        Returns:
            A list (batch) of tuples of (confidence_masks, category_mask).
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Check if options changed to decide whether to reinitialize
        new_options_tuple = self._get_options_tuple(running_mode=running_mode, delegate=delegate, **kwargs)
        current_options_tuple = None
        if self._current_options:
            current_mode = "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video"
            current_delegate = "cpu" if self._current_options.base_options.delegate == python.BaseOptions.Delegate.CPU else "gpu"
            current_options_tuple = self._get_options_tuple(running_mode=current_mode, delegate=current_delegate)

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            if self._detector_instance and hasattr(self._detector_instance, "close"):
                self._detector_instance.close()
            self._detector_instance = self._initialize_detector(running_mode, delegate, **kwargs)

        batch_size = image.shape[0]
        batch_results = []
        is_video_mode = self._current_options.running_mode == vision.RunningMode.VIDEO
        
        # Use segment/segment_for_video methods directly
        segment_func = self._detector_instance.segment_for_video if is_video_mode else self._detector_instance.segment
        
        # Process each image in the batch
        for i in range(batch_size):
            img_tensor = image[i]
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            try:
                if is_video_mode:
                    timestamp_ms = self._timestamp_provider.next()
                    detection_result = segment_func(mp_image, timestamp_ms)
                else:
                    detection_result = segment_func(mp_image)
                
                batch_results.append(self._process_detection_result(detection_result))
            except Exception as e:
                logger.error(f"Error during segmentation for image {i}: {e}")
                # Append None results for this image to maintain batch integrity
                batch_results.append((None, None))

        return batch_results

    def segment(
        self,
        image: torch.Tensor,
        output_confidence_masks: bool = True,
        output_category_mask: bool = False,
        running_mode: str = "video",
        delegate_mode="cpu",
    ):
        """Segments the input image tensor.

        Args:
            image: Input image tensor (BHWC, RGB, 0-1 float).
                   NOTE: This implementation currently only supports batch size 1 for simplicity.
            output_confidence_masks: Whether to output confidence masks (float, 0-1).
            output_category_mask: Whether to output a category mask (uint8).
            running_mode: Processing mode ("image" or "video").
            delegate_mode: Computation delegate ('cpu' or 'gpu').

        Returns:
            A tuple containing:
                - confidence_masks_batch: List (batch) of lists (mask per category) of confidence masks (Torch tensors, HW) or None.
                - category_mask_batch: List (batch) of category masks (Torch tensor, HW) or None.
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")
        if not output_confidence_masks and not output_category_mask:
            raise ValueError("At least one output type (confidence or category mask) must be enabled.")

        # Use the generic detect method from BaseDetector
        batch_results = self.detect(
            image, 
            running_mode=running_mode, 
            delegate=delegate_mode,
            output_confidence_masks=output_confidence_masks,
            output_category_mask=output_category_mask
        )
        
        # Unzip the batch results for backward compatibility
        confidence_masks_batch = []
        category_mask_batch = []
        
        for result_pair in batch_results:
            confidence_masks, category_mask = result_pair
            confidence_masks_batch.append(confidence_masks)
            category_mask_batch.append(category_mask)
        
        return confidence_masks_batch, category_mask_batch

    def close(self):
        """Closes the segmenter instance if it exists."""
        if self._detector_instance:
            logger.info("Closing ImageSegmenter instance.")
            try:
                self._detector_instance.close()
            except Exception as e:
                logger.warning(f"Error closing segmenter instance: {e}")
            self._detector_instance = None
            self._current_options = None

    def __del__(self):
        # Ensure resources are released when the object is destroyed
        self.close()

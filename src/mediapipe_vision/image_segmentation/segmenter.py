"""Image Segmentation implementation using MediaPipe.

This module contains the implementation of MediaPipe Image Segmentation functionality.
"""

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ImageSegmenter
import platform # Needed for OS check
import logging
from ..utils.timestamp_provider import TimestampProvider # Import TimestampProvider

logger = logging.getLogger(__name__)

class ImageSegmenter:
    """Performs image segmentation using MediaPipe ImageSegmenter.
       Adapted from ComfyUI-Stream-Pack implementation.
    """

    def __init__(self, model_path: str):
        """Initialize the segmenter with the model path.

        Args:
            model_path: Path to the MediaPipe ImageSegmenter .tflite or .task file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")

        self.model_path = model_path
        # Keep track of the initialized segmenter instance and its config
        self._detector_instance = None
        self.current_config = None
        self.delegate = None # Store delegate used
        self._timestamp_provider = None # Added

    def _initialize_segmenter(
            self, 
            output_confidence_masks: bool, 
            output_category_mask: bool, 
            running_mode: str, delegate_mode="cpu"):
        """Initializes the ImageSegmenter or returns existing if config matches.
           Handles GPU delegate selection.
        """
        # Return type removed from signature as it's handled internally
        new_config = (output_confidence_masks, output_category_mask, running_mode, delegate_mode)
        
        # Re-initialize if config changed or not initialized
        if self._detector_instance is None or self.current_config != new_config:
            if self._detector_instance:
                 logger.info("Closing existing ImageSegmenter due to config change.")
                 try:
                     self._detector_instance.close()
                 except Exception as e:
                      logger.warning(f"Error closing previous segmenter: {e}")
                 self._detector_instance = None
            
            logger.info(f"Initializing ImageSegmenter: model='{self.model_path}', conf_mask={output_confidence_masks}, cat_mask={output_category_mask}, mode={running_mode}, delegate={delegate_mode}")
            
            # Configure BaseOptions with delegate
            use_gpu = (delegate_mode.lower() == "gpu")
            if use_gpu and platform.system().lower() == "windows":
                logger.warning("GPU delegate is not supported on Windows for Python MediaPipe Tasks. Falling back to CPU.")
                use_gpu = False
                delegate_mode = "cpu"
                
            delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
            base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate)
            
            try:
                mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO
                options = vision.ImageSegmenterOptions(
                    base_options=base_options,
                    running_mode=mode_enum,
                    output_confidence_masks=output_confidence_masks,
                    output_category_mask=output_category_mask
                )
                self._detector_instance = vision.ImageSegmenter.create_from_options(options)
                self.current_config = new_config
                self.delegate = delegate_mode
                self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None
                logger.info("ImageSegmenter initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to create ImageSegmenter: {e}")
                # Raise error to prevent using a failed instance
                raise RuntimeError(f"Failed to initialize MediaPipe ImageSegmenter: {e}")
        else:
            # logger.debug("Reusing existing ImageSegmenter instance.") # Optional: for debugging
            pass
            
        # No explicit return needed, instance is stored in self._detector_instance

    def segment(self, image: torch.Tensor, output_confidence_masks: bool = True, output_category_mask: bool = False, 
                running_mode: str = "video", delegate_mode="cpu"):
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
        # if image.shape[0] != 1: # Temporarily removed batch size 1 restriction
        #     logger.warning("ImageSegmenter currently processes batch > 1 sequentially. Performance may vary.")
            # raise ValueError("Input tensor must be in BHWC format with batch size 1.") 
        if not output_confidence_masks and not output_category_mask:
             raise ValueError("At least one output type (confidence or category mask) must be enabled.")

        # Initialize (or get existing) segmenter instance
        # Initialization now happens within the function if needed
        self._initialize_segmenter(output_confidence_masks, output_category_mask, running_mode, delegate_mode)
        if not self._detector_instance:
             # Initialization failed, return None or raise error
             logger.error("Segmenter instance is not available after initialization attempt.")
             # Return structure matching expected output format (list per batch item)
             return [None] * image.shape[0], [None] * image.shape[0]

        batch_size = image.shape[0]
        batch_results_confidence = [] if output_confidence_masks else None
        batch_results_category = [] if output_category_mask else None

        # Determine the correct segmentation function based on mode
        is_video_mode = self.current_config[2] == "video"
        segment_func = self._detector_instance.segment_for_video if is_video_mode else self._detector_instance.segment

        for i in range(batch_size):
            img_tensor = image[i] # HWC
            # Convert image format
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            # Perform segmentation
            segmentation_result = None
            try:
                if is_video_mode:
                    timestamp_ms = self._timestamp_provider.next()
                    segmentation_result = segment_func(mp_image, timestamp_ms)
                else:
                    segmentation_result = segment_func(mp_image)
            except Exception as e:
                 logger.error(f"Error during segmentation for image {i}: {e}")
                 # Append None results for this image
                 if output_confidence_masks: batch_results_confidence.append(None)
                 if output_category_mask: batch_results_category.append(None)
                 continue # Skip to the next image

            # Process results for the current image
            current_image_confidence_masks = None
            current_image_category_mask = None

            if output_confidence_masks:
                if segmentation_result and segmentation_result.confidence_masks:
                    current_image_confidence_masks = []
                    for mask in segmentation_result.confidence_masks:
                        mask_np = mask.numpy_view()
                        mask_tensor = torch.from_numpy(mask_np).float() # Keep as HW
                        current_image_confidence_masks.append(mask_tensor)
                batch_results_confidence.append(current_image_confidence_masks)

            if output_category_mask:
                if segmentation_result and segmentation_result.category_mask:
                    mask_np = segmentation_result.category_mask.numpy_view()
                    # Category mask is uint8, convert to long tensor
                    current_image_category_mask = torch.from_numpy(mask_np).long() # Keep as HW
                batch_results_category.append(current_image_category_mask)

        return batch_results_confidence, batch_results_category

    def close(self):
         """Closes the segmenter instance if it exists."""
         if self._detector_instance:
            logger.info("Closing ImageSegmenter instance.")
            try:
                 self._detector_instance.close()
            except Exception as e:
                 logger.warning(f"Error closing segmenter instance: {e}")
            self._detector_instance = None
            self.current_config = None
            self.delegate = None

    def __del__(self):
        # Ensure resources are released when the object is destroyed
        self.close()
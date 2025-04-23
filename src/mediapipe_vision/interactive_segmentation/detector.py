import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.interactive_segmenter import (
    InteractiveSegmenter, 
)
from mediapipe.tasks.python.vision import interactive_segmenter
from mediapipe.tasks.python.components.containers import keypoint

from typing import List, Optional
from ..types import PointOfInterest
from ..utils.timestamp_provider import TimestampProvider

import logging
logger = logging.getLogger(__name__)

NormalizedKeypoint = keypoint.NormalizedKeypoint

class InteractiveSegmenterProcessor:
    """Performs interactive image segmentation using MediaPipe."""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        self.model_path = model_path
        self.detector_instance = None
        self.current_options = None
        self._timestamp_provider = None

    def _initialize_processor(self, output_category_mask: bool, output_confidence_mask: bool):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = interactive_segmenter.InteractiveSegmenterOptions(
             base_options=base_options,
             output_category_mask=output_category_mask,
             output_confidence_masks=output_confidence_mask
         )
        self.current_options = (output_category_mask, output_confidence_mask)
        self._timestamp_provider = TimestampProvider()
        return interactive_segmenter.InteractiveSegmenter.create_from_options(options)

    def segment(self, image: torch.Tensor, points_of_interest: List[PointOfInterest], 
                output_category_mask: bool = True, output_confidence_mask: bool = False) -> List[Optional[torch.Tensor]]:
        """Performs interactive segmentation based on points of interest.
        Iterates through points, generates mask for each, and combines them.

        Args:
            image: Input image tensor (BHWC).
            points_of_interest: List of PointOfInterest objects (normalized coords).
            output_category_mask: If True, outputs a category mask.
            output_confidence_mask: If True, outputs confidence masks.

        Returns:
            A list (batch) containing a single mask tensor (HW) or None per image.
            Priority is given to category mask if both outputs requested.
        """
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")
        if not points_of_interest:
             raise ValueError("At least one point_of_interest must be provided.")

        batch_size = image.shape[0]
        batch_results_combined = []
        
        new_options = (output_category_mask, output_confidence_mask)
        if self.detector_instance is None or self.current_options != new_options:
             if self.detector_instance and hasattr(self.detector_instance, 'close'):
                 self.detector_instance.close()
             self.detector_instance = self._initialize_processor(output_category_mask, output_confidence_mask)

        for i in range(batch_size):
            img_tensor = image[i] # HWC
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            individual_masks_for_image = []

            for point in points_of_interest:
                roi_keypoint_single = NormalizedKeypoint(x=point.x, y=point.y, label=point.label)
                roi = interactive_segmenter.RegionOfInterest(
                    format=interactive_segmenter.RegionOfInterest.Format.KEYPOINT,
                    keypoint=roi_keypoint_single
                )
                
                segmentation_result = self.detector_instance.segment(mp_image, roi)
                
                current_point_mask = None
                if output_category_mask and segmentation_result.category_mask:
                    mask_np = segmentation_result.category_mask.numpy_view()
                    current_point_mask = torch.from_numpy(mask_np).float()
                elif output_confidence_mask and segmentation_result.confidence_masks:
                    if segmentation_result.confidence_masks:
                        mask_np = segmentation_result.confidence_masks[0].numpy_view()
                        current_point_mask = torch.from_numpy(mask_np).float()
                
                if current_point_mask is not None:
                    individual_masks_for_image.append(current_point_mask)
                else:
                    logger.warning(f"Segmentation failed for point {point} on image {i}. Skipping point.")
            combined_mask_for_image = None
            if not individual_masks_for_image:
                h, w = img_tensor.shape[0], img_tensor.shape[1]
                combined_mask_for_image = torch.zeros((h, w), dtype=torch.float32, device=image.device)
                logger.warning(f"No successful segmentations for any points on image {i}. Returning zero mask.")
            elif len(individual_masks_for_image) == 1:
                combined_mask_for_image = individual_masks_for_image[0]
            else:
                mask_stack = torch.stack(individual_masks_for_image, dim=0)
                combined_mask_for_image = torch.max(mask_stack, dim=0).values
            
            batch_results_combined.append(combined_mask_for_image)
        return batch_results_combined 
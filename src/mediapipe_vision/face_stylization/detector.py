from typing import Any, List

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider
from ..common import BaseDetector


class FaceStylizer(BaseDetector[torch.Tensor]):
    """Applies stylization to faces using MediaPipe FaceStylizer."""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        self.model_path = model_path
        self._detector_instance = None
        self.current_options = None
        self._timestamp_provider = None

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.FaceStylizerOptions:
        """Create FaceStylizer-specific options.
        Note: FaceStylizer only supports IMAGE mode, not VIDEO mode.
        """
        # FaceStylizerOptions has no specific config other than base options
        # and doesn't support VIDEO mode - we ignore the mode_enum parameter
        return vision.FaceStylizerOptions(base_options=base_options)

    def _create_detector_instance(self, options: vision.FaceStylizerOptions) -> vision.FaceStylizer:
        return vision.FaceStylizer.create_from_options(options)

    def _get_options_tuple(self, running_mode: str = None, delegate: str = None, **kwargs) -> tuple:
        # FaceStylizer has no additional options to track
        return (delegate,)

    def _process_detection_result(self, stylization_result: Any) -> torch.Tensor:
        """Process stylization result to torch tensor."""
        if stylization_result:
            # Convert MediaPipe Image back to NumPy (HWC, uint8)
            stylized_np = stylization_result.numpy_view()
            # Convert to float tensor (HWC, 0-1)
            return torch.from_numpy(stylized_np).float().div(255.0)
        return None  # If no result, we'll handle in the stylize method

    def stylize(self, image: torch.Tensor) -> torch.Tensor:
        """Applies face stylization to the input image tensor(s)."""
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        # Ensure detector is initialized - always use IMAGE mode
        if self._detector_instance is None:
            self._initialize_detector("image", "cpu")

        batch_size = image.shape[0]
        batch_results = []

        for i in range(batch_size):
            img_tensor = image[i]  # HWC
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            # FaceStylizer only has the stylize method, not detect
            stylization_result = self._detector_instance.stylize(mp_image)
            output_image_tensor = self._process_detection_result(stylization_result)
            
            # If processing failed (no face detected), return original image
            if output_image_tensor is None:
                output_image_tensor = img_tensor
                
            batch_results.append(output_image_tensor)

        output_batch_tensor = torch.stack(batch_results, dim=0)
        return output_batch_tensor

    def _initialize_detector(self, running_mode: str, delegate: str):
        """Initializes the FaceStylizer."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        # FaceStylizerOptions has no specific config other than base options
        # Check if FaceStylizer supports VIDEO mode
        # According to docs (as of late 2023/early 2024), it seems it only supports IMAGE mode.
        # Sticking with IMAGE mode.
        options = self._create_detector_options(base_options, vision.RunningMode.IMAGE)
        self.current_options = ()
        self._timestamp_provider = TimestampProvider()
        self._detector_instance = self._create_detector_instance(options)

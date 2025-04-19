import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Optional
from ..utils.timestamp_provider import TimestampProvider

class FaceStylizer:
    """Applies stylization to faces using MediaPipe FaceStylizer."""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        self.model_path = model_path
        self.detector_instance = None
        self.current_options = None
        self._timestamp_provider = None

    def _initialize_detector(self):
        """Initializes the FaceStylizer."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        # FaceStylizerOptions has no specific config other than base options
        # Check if FaceStylizer supports VIDEO mode
        # According to docs (as of late 2023/early 2024), it seems it only supports IMAGE mode.
        # Sticking with IMAGE mode.
        options = vision.FaceStylizerOptions(
            base_options=base_options
            # running_mode=vision.RunningMode.VIDEO # Not supported
            )
        self.current_options = ()
        self._timestamp_provider = TimestampProvider()
        return vision.FaceStylizer.create_from_options(options)

    def stylize(self, image: torch.Tensor) -> torch.Tensor: # Output is IMAGE
        """Applies face stylization to the input image tensor(s)."""
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")

        if self.detector_instance is None:
            self.detector_instance = self._initialize_detector()

        batch_size = image.shape[0]
        batch_results = []

        for i in range(batch_size):
            img_tensor = image[i] # HWC
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            stylization_result = self.detector_instance.stylize(mp_image)
            
            output_image_tensor = None
            if stylization_result:
                 # Convert MediaPipe Image back to NumPy (HWC, uint8)
                 stylized_np = stylization_result.numpy_view()
                 # Convert to float tensor (HWC, 0-1)
                 output_image_tensor = torch.from_numpy(stylized_np).float().div(255.0)
            else:
                 # Handle cases where stylization fails (e.g., no face detected)
                 # Return original image tensor for this batch item
                 output_image_tensor = img_tensor
            
            batch_results.append(output_image_tensor)

        output_batch_tensor = torch.stack(batch_results, dim=0)
        return output_batch_tensor 
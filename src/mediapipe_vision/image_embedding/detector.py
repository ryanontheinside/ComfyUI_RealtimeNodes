from typing import List

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Import TimestampProvider from the new location
from ...utils.timing import TimestampProvider

# Import BaseDetector class
from ..common.base_detector import BaseDetector

# Import new types
from ..types import ImageEmbedderResult


class ImageEmbedder(BaseDetector):
    """Generates image embeddings using MediaPipe ImageEmbedder."""

    def _create_detector_options(self, base_options: python.BaseOptions, 
                              mode_enum: vision.RunningMode, **kwargs) -> vision.ImageEmbedderOptions:
        """Create image embedder-specific options.
        
        Args:
            base_options: MediaPipe base options with model path and delegate.
            mode_enum: RunningMode enum (IMAGE or VIDEO).
            **kwargs: Detector-specific configuration parameters.
            
        Returns:
            Configured ImageEmbedderOptions object.
        """
        l2_normalize = kwargs.get('l2_normalize', True)
        quantize = kwargs.get('quantize', False)
        
        return vision.ImageEmbedderOptions(
            base_options=base_options,
            running_mode=mode_enum,
            l2_normalize=l2_normalize,
            quantize=quantize,
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
        l2_normalize = kwargs.get('l2_normalize', True)
        quantize = kwargs.get('quantize', False)
        
        return (l2_normalize, quantize, running_mode, delegate.lower() if delegate else "cpu")

    def _process_detection_result(self, detection_result) -> List[ImageEmbedderResult]:
        """Process embedding results.
        
        Args:
            detection_result: Raw embedding result from MediaPipe.
            
        Returns:
            List of ImageEmbedderResult objects.
        """
        current_image_embeddings = []
        
        # Check for quantize parameter in current options
        quantize = False
        if self._current_options and hasattr(self._current_options, 'quantize'):
            quantize = self._current_options.quantize
            
        if detection_result and detection_result.embeddings:
            for emb in detection_result.embeddings:
                result = ImageEmbedderResult(
                    float_embedding=list(emb.embedding) if not quantize else None,
                    quantized_embedding=bytes(emb.embedding) if quantize else None,
                    head_index=emb.head_index,
                    head_name=emb.head_name,
                )
                current_image_embeddings.append(result)
                
        return current_image_embeddings

    def _create_detector_instance(self, options: vision.ImageEmbedderOptions) -> vision.ImageEmbedder:
        """Create embedder instance.
        
        Args:
            options: Configured ImageEmbedderOptions.
            
        Returns:
            MediaPipe ImageEmbedder instance.
        """
        return vision.ImageEmbedder.create_from_options(options)
        
    def detect(self, image: torch.Tensor, running_mode: str = "video", 
              delegate: str = "cpu", **kwargs) -> List[List[ImageEmbedderResult]]:
        """Override of the BaseDetector detect method to use embed/embed_for_video.
        
        Args:
            image: Input image tensor in BHWC format.
            running_mode: "image" or "video".
            delegate: "cpu" or "gpu".
            **kwargs: Detector-specific parameters.
            
        Returns:
            A list (batch) of lists (detections per image) of detection results.
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
        
        # Use embed/embed_for_video methods directly
        embed_func = self._detector_instance.embed_for_video if is_video_mode else self._detector_instance.embed
        
        # Process each image in the batch
        for i in range(batch_size):
            img_tensor = image[i]
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                timestamp_ms = self._timestamp_provider.next()
                detection_result = embed_func(mp_image, timestamp_ms)
            else:
                detection_result = embed_func(mp_image)

            batch_results.append(self._process_detection_result(detection_result))

        return batch_results

    def embed(
        self,
        image: torch.Tensor,
        l2_normalize: bool = True,
        quantize: bool = False,
        running_mode: str = "video",
        delegate: str = "cpu",
    ) -> List[List[ImageEmbedderResult]]:
        """Generates embeddings for the input image tensor(s).
        
        Args:
            image: Input image tensor (BHWC, RGB, 0-1 float).
            l2_normalize: Whether to L2-normalize the embedding vectors.
            quantize: Whether to quantize the embedding vectors.
            running_mode: Processing mode ("image" or "video").
            delegate: Computation delegate ('cpu' or 'gpu').
            
        Returns:
            A list (batch) of lists (embeddings) of ImageEmbedderResult objects.
        """
        # Use the generic detect method from BaseDetector
        return self.detect(
            image, 
            running_mode=running_mode, 
            delegate=delegate,
            l2_normalize=l2_normalize,
            quantize=quantize
        )

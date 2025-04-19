import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import List, Optional
# Import new types
from ..types import ImageEmbedderResult
from ..utils.timestamp_provider import TimestampProvider # Import TimestampProvider

class ImageEmbedder:
    """Generates image embeddings using MediaPipe ImageEmbedder."""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("A valid model_path must be provided.")
        self.model_path = model_path
        self._detector_instance = None
        self._current_options = None
        self._timestamp_provider = None # Added

    def _initialize_detector(self, l2_normalize: bool, quantize: bool, 
                             running_mode: str, delegate: str): # Added running_mode
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO # Added
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == 'cpu' else BaseOptions.Delegate.GPU
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = vision.ImageEmbedderOptions(
            base_options=base_options,
            running_mode=mode_enum, # Use enum mode
            l2_normalize=l2_normalize,
            quantize=quantize
        )
        self._current_options = options
        # Only create timestamp provider for VIDEO mode
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None 
        return vision.ImageEmbedder.create_from_options(options)

    def embed(self, image: torch.Tensor, l2_normalize: bool = True, quantize: bool = False, 
                running_mode: str = "video", delegate: str = 'cpu') -> List[List[ImageEmbedderResult]]: # Added running_mode
        """Generates embeddings for the input image tensor(s)."""
        if image.dim() != 4:
            raise ValueError("Input tensor must be in BHWC format.")
            
        # Add running_mode to tuple check
        new_options_tuple = (l2_normalize, quantize, running_mode, delegate.lower())
        current_options_tuple = None
        if self._current_options:
             current_options_tuple = (
                 self._current_options.l2_normalize,
                 self._current_options.quantize,
                 "image" if self._current_options.running_mode == vision.RunningMode.IMAGE else "video", # Map enum back to string
                 'cpu' if self._current_options.base_options.delegate == BaseOptions.Delegate.CPU else 'gpu'
             )

        if self._detector_instance is None or new_options_tuple != current_options_tuple:
            # ImageEmbedder doesn't seem to have close(), so just replace instance
            if self._detector_instance and hasattr(self._detector_instance, 'close'):
                 self._detector_instance.close()
            self._detector_instance = self._initialize_detector(l2_normalize, quantize, running_mode, delegate)

        batch_size = image.shape[0]
        batch_results = []
        # Determine the correct embedding function based on mode
        is_video_mode = self._current_options.running_mode == vision.RunningMode.VIDEO
        embed_func = self._detector_instance.embed_for_video if is_video_mode else self._detector_instance.embed

        for i in range(batch_size):
            img_tensor = image[i] 
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                timestamp_ms = self._timestamp_provider.next()
                embedding_result = embed_func(mp_image, timestamp_ms)
            else:
                embedding_result = embed_func(mp_image)
            
            current_image_embeddings = []
            if embedding_result and embedding_result.embeddings: # Added check for embedding_result
                for emb in embedding_result.embeddings:
                    result = ImageEmbedderResult(float_embedding=list(emb.embedding) if not quantize else None, quantized_embedding=bytes(emb.embedding) if quantize else None, head_index=emb.head_index, head_name=emb.head_name)
                    current_image_embeddings.append(result)
            batch_results.append(current_image_embeddings)
            
        return batch_results 
import torch
import torch.nn.functional as F
import logging
from typing import List, Optional

from ...src.types import ImageEmbedderResult # Assuming this is the relevant type

logger = logging.getLogger(__name__)

class ReshapeMediaPipeEmbeddingNode:
    """Reshapes MediaPipe embedding for experimental use as ComfyUI Conditioning.
       WARNING: Semantically different from standard CLIP conditioning.
       Pads/truncates the embedding vector and replicates it across 77 tokens.
    """
    CATEGORY = "MediaPipeVision/ImageEmbedding/_Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_embeddings": ("IMAGE_EMBEDDINGS",), # List[List[ImageEmbedderResult]]
                "target_dimension": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8, 
                                          "tooltip": "Target dimension D for the reshaped (1, 77, D) tensor (e.g., 768 for SD1.5, 1024/1280 for SDXL)"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "reshape_embedding"

    def reshape_embedding(self, image_embeddings: List[List[ImageEmbedderResult]], target_dimension: int):
        if not image_embeddings or not image_embeddings[0]:
            raise ValueError("Input image_embeddings are empty.")
            
        # --- Process First Embedding in Batch/Head ---
        # Assuming we use the first embedding result from the first image in the batch
        first_embedding_result = image_embeddings[0][0]
        
        if first_embedding_result.float_embedding:
            embedding_vector = first_embedding_result.float_embedding
        elif first_embedding_result.quantized_embedding:
            # Note: Using quantized embedding might require dequantization logic
            # For now, raise error or warn, as direct reshaping is less meaningful
            raise NotImplementedError("Reshaping quantized embeddings is not directly supported. Use float embeddings.")
        else:
            raise ValueError("No valid embedding found in the input.")
            
        # Convert to tensor
        embed_tensor = torch.tensor(embedding_vector, dtype=torch.float32)
        orig_dim = embed_tensor.shape[0]

        # --- Pad or Truncate --- 
        if orig_dim < target_dimension:
            # Pad with zeros
            padding = torch.zeros(target_dimension - orig_dim, dtype=torch.float32)
            final_vector = torch.cat((embed_tensor, padding), dim=0)
        elif orig_dim > target_dimension:
            # Truncate
            final_vector = embed_tensor[:target_dimension]
        else:
            final_vector = embed_tensor
            
        # --- Reshape and Replicate for Conditioning --- 
        # Reshape to (1, 1, D)
        cond_vec = final_vector.unsqueeze(0).unsqueeze(0)
        # Replicate to (1, 77, D)
        cond_batch = cond_vec.repeat(1, 77, 1)

        # --- Create Pooled Output --- 
        # Use the padded/truncated vector as the pooled output (1, D)
        pooled_output = final_vector.unsqueeze(0)

        # --- Package as CONDITIONING --- 
        # Standard ComfyUI conditioning format: List[List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]
        conditioning = [[cond_batch, {"pooled_output": pooled_output}]]

        logger.warning("Reshaped MediaPipe embedding to CONDITIONING format. Use experimentally.")
        return (conditioning,)

# --- Comparison Node --- 
class CompareMediaPipeEmbeddingsNode:
    """Calculates the cosine similarity between two MediaPipe image embeddings."""
    CATEGORY = "MediaPipeVision/ImageEmbedding"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_embeddings_1": ("IMAGE_EMBEDDINGS",), 
                "image_embeddings_2": ("IMAGE_EMBEDDINGS",), 
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("cosine_similarity",)
    FUNCTION = "compare_embeddings"

    def compare_embeddings(self, image_embeddings_1: List[List[ImageEmbedderResult]], 
                             image_embeddings_2: List[List[ImageEmbedderResult]]):
        
        def _extract_vector(embeddings: List[List[ImageEmbedderResult]]) -> Optional[torch.Tensor]:
            """Helper to extract the first float embedding vector."""
            if not embeddings or not embeddings[0]:
                return None
            first_result = embeddings[0][0]
            if first_result.float_embedding:
                return torch.tensor(first_result.float_embedding, dtype=torch.float32)
            else:
                logger.warning("Cannot compare non-float embeddings.")
                return None

        vec1 = _extract_vector(image_embeddings_1)
        vec2 = _extract_vector(image_embeddings_2)

        if vec1 is None or vec2 is None:
            logger.error("Could not extract valid float embedding vectors for comparison.")
            return (0.0,) # Return 0 similarity on error
            
        # Ensure vectors have the same dimension (pad/truncate shorter one)
        dim1, dim2 = vec1.shape[0], vec2.shape[0]
        if dim1 != dim2:
            logger.warning(f"Embedding dimensions differ ({dim1} vs {dim2}). Padding/truncating for comparison.")
            max_dim = max(dim1, dim2)
            if dim1 < max_dim:
                vec1 = F.pad(vec1, (0, max_dim - dim1))
            elif dim2 < max_dim:
                vec2 = F.pad(vec2, (0, max_dim - dim2))
        
        # Calculate Cosine Similarity
        # Normalize vectors before dot product for cosine similarity
        similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)
        
        return (similarity.item(),)


# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "ReshapeMediaPipeEmbedding": ReshapeMediaPipeEmbeddingNode,
    "CompareMediaPipeEmbeddings": CompareMediaPipeEmbeddingsNode, # Add new node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReshapeMediaPipeEmbedding": "Reshape MediaPipe Embedding (Experimental)",
    "CompareMediaPipeEmbeddings": "Compare MediaPipe Embeddings", # Add new node name
} 
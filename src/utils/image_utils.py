import torch
import numpy as np
import cv2

def convert_to_cv2(tensor: torch.Tensor) -> list:
    """Converts a ComfyUI IMAGE tensor (BHWC, float32, 0-1) to a list of cv2 images (BGR, uint8)."""
    if tensor.ndim != 4 or tensor.shape[3] != 3:
        raise ValueError(f"Input tensor must be BHWC with 3 channels, got shape: {tensor.shape}")
    
    # Convert entire batch to numpy, scale to 0-255, and convert to uint8
    images_np = tensor.cpu().numpy()
    images_np = (images_np * 255).clip(0, 255).astype(np.uint8)
    
    # Convert RGB to BGR for cv2
    cv2_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
    return cv2_images

def convert_to_tensor(cv2_images) -> torch.Tensor:
    """Converts cv2 image(s) (BGR, uint8) to a ComfyUI IMAGE tensor (BHWC, float32, 0-1).
    
    Args:
        cv2_images: A single cv2 image or a list of cv2 images
    """
    # Handle both single image and list of images
    if not isinstance(cv2_images, list):
        cv2_images = [cv2_images]
    
    # Process each image
    np_images = []
    for cv2_image in cv2_images:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Scale to 0-1
        image_np = rgb_image.astype(np.float32) / 255.0
        np_images.append(image_np)
    
    # Stack all images into a batch
    batch_np = np.stack(np_images, axis=0)
    
    # Convert to tensor (already has batch dimension)
    tensor = torch.from_numpy(batch_np)
    
    return tensor
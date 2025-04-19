# --- Helper Function --- 
import torch
import numpy as np
import cv2
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def create_mask_from_points(height: int, width: int, points: List[Tuple[int, int]], device='cpu') -> torch.Tensor:
    """Creates a binary mask tensor from a list of points using convex hull."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        logger.warning(f"Cannot create mask from {len(points)} points (need >= 3 for convex hull). Returning empty mask.")
        return torch.from_numpy(mask).float().unsqueeze(0).to(device)
        
    try:
        np_points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(np_points)
        cv2.fillPoly(mask, [hull], 1)
    except Exception as e:
        logger.error(f"Error creating mask polygon: {e}")
        return torch.from_numpy(mask).float().unsqueeze(0).to(device)
        
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(device)
    return mask_tensor
# --- Helper Function --- 
import torch
import numpy as np
import cv2
from typing import List, Tuple
import logging
from ...utils.image import create_mask_from_points

logger = logging.getLogger(__name__)

# create_mask_from_points is now imported from utils.image
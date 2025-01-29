import torch
import comfy.utils
import numpy as np
import base64
import re
from io import BytesIO
from PIL import Image
import nodes
import random
from torchvision import transforms

MAX_RESOLUTION = nodes.MAX_RESOLUTION  # Get the same max resolution as core nodes

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class DTypeConverter:
    """Converts masks to specified data types"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # Explicitly accept only MASK input
                "dtype": (["float16", "uint8", "float32", "float64"],),
            }
        }

    CATEGORY = "utils"
    RETURN_TYPES = ("MASK",)  # Return only MASK type
    FUNCTION = "convert_dtype"

    DTYPE_MAP = {
        "uint8": torch.uint8,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64
    }

    def convert_dtype(self, mask, dtype):
        target_dtype = self.DTYPE_MAP[dtype]
        
        if target_dtype == torch.uint8:
            if mask.is_floating_point():
                converted = (mask * 255.0).round().to(torch.uint8)
            else:
                converted = (mask > 0).to(torch.uint8) * 255
        else:  # Converting to float
            if mask.dtype == torch.uint8:
                converted = (mask > 0).to(target_dtype)
            else:
                converted = mask.to(target_dtype)
        
        return (converted,)   

class FastWebcamCapture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("WEBCAM", {}),
                "width": ("INT", {"default": 640, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 480, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "capture_on_queue": ("BOOLEAN", {"default": True}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_capture"
    
    CATEGORY = "image"

    def process_capture(self, image, width, height, capture_on_queue):
        # Check if we got a data URL
        if isinstance(image, str) and image.startswith('data:image/'):
            # Extract the base64 data after the comma
            base64_data = re.sub('^data:image/.+;base64,', '', image)
            
            # Convert base64 to PIL Image
            buffer = BytesIO(base64.b64decode(base64_data))
            pil_image = Image.open(buffer).convert("RGB")
            
            # Convert PIL to numpy array
            image = np.array(pil_image)
            
            # Handle resize if requested
            if width > 0 and height > 0:
                import cv2
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            
            # Convert to float32 and normalize to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension and convert to torch tensor
            # ComfyUI expects BHWC format
            image = torch.from_numpy(image)[None,...]
            
            return (image,)
        else:
            raise ValueError("Invalid image format received from webcam")

coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class YOLOSimilarityCompare:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ULTRALYTICS_RESULTS1": ("ULTRALYTICS_RESULTS",),
                "ULTRALYTICS_RESULTS2": ("ULTRALYTICS_RESULTS",),
                "class_weight": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Weight for class similarity - how much to consider matching object types"
                }),
                "spatial_weight": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Weight for spatial similarity - how much to consider object positions"
                }),
                "confidence_weight": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Weight for confidence similarity - how much to consider detection confidence"
                }),
                "size_weight": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Weight for size similarity - how much to consider matching object sizes"
                }),
                "relationship_weight": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Weight for relationship similarity - how much to consider distances between objects"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Threshold for similarity - scores above this value will return True"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING",)
    RETURN_NAMES = ("similarity_score", "above_threshold", "explanation")
    FUNCTION = "compare_images"
    CATEGORY = "ScavengerHunt"

    def compute_size_similarity(self, boxes1, boxes2):
        if len(boxes1) == 0 or len(boxes2) == 0:
            return 0.0
        
        # Compute areas
        areas1 = (boxes1[:, 2] * boxes1[:, 3]).cpu().numpy()  # width * height
        areas2 = (boxes2[:, 2] * boxes2[:, 3]).cpu().numpy()
        
        # Normalize areas by image size
        norm_areas1 = areas1 / np.max(areas1) if len(areas1) > 0 else areas1
        norm_areas2 = areas2 / np.max(areas2) if len(areas2) > 0 else areas2
        
        # Compare average sizes
        avg_size1 = np.mean(norm_areas1)
        avg_size2 = np.mean(norm_areas2)
        
        return 1.0 - abs(avg_size1 - avg_size2)

    def compute_relationship_similarity(self, boxes1, boxes2):
        if len(boxes1) < 2 or len(boxes2) < 2:
            return 0.0
        
        def compute_pairwise_distances(boxes):
            centers = boxes[:, :2]  # Get centers (x, y)
            distances = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)
            return np.array(distances)
        
        # Compute normalized pairwise distances
        dist1 = compute_pairwise_distances(boxes1.cpu().numpy())
        dist2 = compute_pairwise_distances(boxes2.cpu().numpy())
        
        # Normalize distances
        if len(dist1) > 0:
            dist1 = dist1 / np.max(dist1)
        if len(dist2) > 0:
            dist2 = dist2 / np.max(dist2)
        
        # Compare average normalized distances
        avg_dist1 = np.mean(dist1) if len(dist1) > 0 else 0
        avg_dist2 = np.mean(dist2) if len(dist2) > 0 else 0
        
        return 1.0 - abs(avg_dist1 - avg_dist2)

    def compare_images(self, ULTRALYTICS_RESULTS1, ULTRALYTICS_RESULTS2, class_weight=0.3, 
                      spatial_weight=0.2, confidence_weight=0.2, size_weight=0.15, 
                      relationship_weight=0.15, threshold=0.5):
        # Get detections from first image
        boxes1 = ULTRALYTICS_RESULTS1[0].boxes
        classes1 = boxes1.cls.cpu().numpy()
        conf1 = boxes1.conf.cpu().numpy()
        xyxy1 = boxes1.xyxy.cpu().numpy()
        
        # Get detections from second image
        boxes2 = ULTRALYTICS_RESULTS2[0].boxes
        classes2 = boxes2.cls.cpu().numpy()
        conf2 = boxes2.conf.cpu().numpy()
        xyxy2 = boxes2.xyxy.cpu().numpy()
        
        # 1. Class similarity - what objects appear in both images
        unique_classes1 = set(classes1)
        unique_classes2 = set(classes2)
        class_overlap = len(unique_classes1.intersection(unique_classes2)) / len(unique_classes1.union(unique_classes2)) if unique_classes1 or unique_classes2 else 0
        
        # 2. Spatial similarity - compare object locations
        spatial_sim = 0
        if len(xyxy1) > 0 and len(xyxy2) > 0:
            # Normalize coordinates to 0-1 range
            img_size1 = ULTRALYTICS_RESULTS1[0].boxes.orig_shape
            img_size2 = ULTRALYTICS_RESULTS2[0].boxes.orig_shape
            
            norm_boxes1 = xyxy1.copy()
            norm_boxes1[:,[0,2]] /= img_size1[1]  # normalize x by width
            norm_boxes1[:,[1,3]] /= img_size1[0]  # normalize y by height
            
            norm_boxes2 = xyxy2.copy()
            norm_boxes2[:,[0,2]] /= img_size2[1]
            norm_boxes2[:,[1,3]] /= img_size2[0]
            
            # Compare average box positions
            center1 = np.mean(norm_boxes1, axis=0)
            center2 = np.mean(norm_boxes2, axis=0)
            spatial_sim = 1 - np.mean(np.abs(center1 - center2))
        
        # 3. Confidence similarity - compare detection confidences
        conf_sim = 1 - abs(np.mean(conf1) - np.mean(conf2)) if len(conf1) > 0 and len(conf2) > 0 else 0
        
        # 4. Size similarity - compare object sizes
        size_sim = self.compute_size_similarity(boxes1.xywh, boxes2.xywh)
        
        # 5. Relationship similarity - compare distances between objects
        relationship_sim = self.compute_relationship_similarity(boxes1.xywh, boxes2.xywh)
        
        # Weighted combination
        similarity = (class_weight * class_overlap + 
                     spatial_weight * spatial_sim + 
                     confidence_weight * conf_sim +
                     size_weight * size_sim +
                     relationship_weight * relationship_sim)
        
        # Check if above threshold
        above_threshold = similarity >= threshold
        
        # Generate explanation
        explanation = f"""Similarity Score: {similarity:.3f} (Threshold: {threshold})
Above Threshold: {above_threshold}

Class Overlap: {class_overlap:.3f} (weight: {class_weight})
Spatial Similarity: {spatial_sim:.3f} (weight: {spatial_weight})
Confidence Similarity: {conf_sim:.3f} (weight: {confidence_weight})
Size Similarity: {size_sim:.3f} (weight: {size_weight})
Relationship Similarity: {relationship_sim:.3f} (weight: {relationship_weight})

Detected Classes in Image 1: {[coco_classes[int(c)] for c in unique_classes1]}
Detected Classes in Image 2: {[coco_classes[int(c)] for c in unique_classes2]}"""
        
        return (float(similarity), above_threshold, explanation,)


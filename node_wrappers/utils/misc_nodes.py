import torch
import comfy.utils
import numpy as np
import base64
import re
import math
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw, ImageOps
import nodes
import random
from torchvision import transforms
from ...src.utils.general_utils import AlwaysEqualProxy
MAX_RESOLUTION = nodes.MAX_RESOLUTION  # Get the same max resolution as core nodes

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

class QuickShapeMask:
    """A node that quickly generates basic shape masks"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": (["circle", "square"], {
                    "default": "circle",
                    "tooltip": "The shape of the mask to generate"
                }),
                "width": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Width of the shape in pixels"
                }),
                "height": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Height of the shape in pixels"
                }),
                "x": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "X position of the shape center (0 = left edge)"
                }),
                "y": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Y position of the shape center (0 = top edge)"
                }),
                "canvas_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Width of the output mask"
                }),
                "canvas_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Height of the output mask"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of identical masks to generate"
                })
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_mask"
    CATEGORY = "mask"
    
    DESCRIPTION = "Generates a mask containing a basic shape (circle or square) with high performance"

    def generate_mask(self, shape, width, height, x, y, canvas_width, canvas_height, batch_size):
        # Create empty mask
        mask = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        # Calculate boundaries
        half_width = width // 2
        half_height = height // 2
        
        # Calculate shape boundaries
        left = max(0, x - half_width)
        right = min(canvas_width, x + half_width)
        top = max(0, y - half_height)
        bottom = min(canvas_height, y + half_height)
        
        if shape == "square":
            # Simple square mask
            mask[top:bottom, left:right] = 1.0
            
        else:  # circle
            # Create coordinate grids for the region of interest
            Y, X = np.ogrid[top:bottom, left:right]
            
            # Calculate distances from center for the region
            dist_x = (X - x)
            dist_y = (Y - y)
            
            # Create circle mask using distance formula
            circle_mask = (dist_x**2 / (width/2)**2 + dist_y**2 / (height/2)**2) <= 1
            
            # Apply circle to the region
            mask[top:bottom, left:right][circle_mask] = 1.0
        
        # Convert to torch tensor and add batch dimension (BHW format)
        mask_tensor = torch.from_numpy(mask)
        
        # Expand to requested batch size
        mask_tensor = mask_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        return (mask_tensor,)

class TextRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"),),  # Accept any input type and convert to string
                "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "font_size": ("INT", {"default": 48, "min": 1, "max": 512, "step": 1}),
                "font_color": ("STRING", {"default": "white"}),
                "background_color": ("STRING", {"default": "black"}),
                "x_offset": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "align": (["left", "center", "right"], {"default": "center"}),
                "wrap_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8, 
                             "tooltip": "Width to wrap text at (0 = no wrapping)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "render_text"
    CATEGORY = "real-time"

    def render_text(self, any, width, height, font_size, font_color, background_color, 
                   x_offset, y_offset, align, wrap_width):
        # Convert input to string if it isn't already
        any = str(any)
        
        # Create image with background
        image = Image.new('RGBA', (width, height), background_color)
        mask = Image.new('L', (width, height), 0)  # Black mask initially
        
        # Load font from assets directory
        import os
        # Construct the path relative to the main package directory
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) # Go up 3 levels
        font_path = os.path.join(base_path, "fonts", "dejavu-sans", "DejaVuSans.ttf")
        try:
            pil_font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Failed to load font from {font_path}, falling back to default. Error: {str(e)}")
            pil_font = ImageFont.load_default()
        
        # Create drawing objects
        draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)
        
        import textwrap
        
        # Handle text wrapping
        if wrap_width > 0:
            # Convert wrap_width from pixels to approximate character count
            # Assuming average character width is font_size/2 pixels
            char_width = font_size/2
            char_count = max(1, int(wrap_width / char_width))
            any = "\n".join(textwrap.wrap(any, width=char_count))
        
        # Calculate text size
        bbox = draw.textbbox((0, 0), any, font=pil_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        if align == "left":
            x = 0 + x_offset
        elif align == "right":
            x = width - text_width + x_offset
        else:  # center
            x = (width - text_width) // 2 + x_offset
        
        y = (height - text_height) // 2 + y_offset
        
        # Draw text
        draw.multiline_text((x, y), any, font=pil_font, fill=font_color, align=align)
        mask_draw.multiline_text((x, y), any, font=pil_font, fill=255, align=align)
        
        # Convert PIL to tensor
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
        mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0)[None,]
        
        # If image has alpha channel, use it to modify the mask
        if image_tensor.shape[-1] == 4:
            mask_tensor = mask_tensor * image_tensor[..., 3]
            image_tensor = image_tensor[..., :3]  # Keep only RGB channels
        
        return (image_tensor, mask_tensor)

class MultilineText:
    """A simple node for creating and formatting multiline text"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Line 1\nLine 2\nLine 3",
                    "tooltip": "Enter text (use \\n for new lines)"
                }),
                "strip_whitespace": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove leading/trailing whitespace from each line"
                }),
                "remove_empty_lines": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove empty or whitespace-only lines"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_text"
    CATEGORY = "text"
    
    def process_text(self, text, strip_whitespace, remove_empty_lines):
        # Split text into lines
        lines = text.split('\n')
        
        # Process lines according to settings
        if strip_whitespace:
            lines = [line.strip() for line in lines]
        
        if remove_empty_lines:
            lines = [line for line in lines if line.strip()]
        
        # Rejoin lines
        result = '\n'.join(lines)
        
        return (result,)

class LoadImageFromPath_:
    """A simple node that loads an image from a given file path"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image_path": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Full path to the image file to load"
            })
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image_path):
        # Open and process the image
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        # Convert to RGB if needed
        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        image = img.convert("RGB")
        
        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        # Create mask from alpha channel if it exists
        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        
        mask = mask.unsqueeze(0)
        
        return (image, mask)

class FeedbackEffect:
    """
    Ultra-fast real-time visual feedback processor designed for SetState/GetState loops.
    Creates stunning, hypnotic visuals with optimized GPU-accelerated transformations.
    
    For best results: connect to a SetState/GetState loop and adjust parameters in real-time.
    """
    CATEGORY = "image/effects"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_effect"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (typically from GetStateNode)"}),
                "effect": (["fractal_zoom", "kaleidoscope", "plasma_field", "quantum_wave", 
                           "neural_dream", "digital_flow", "hypershock"], {
                    "default": "fractal_zoom",
                    "tooltip": "Type of real-time effect to apply"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.1, 
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Effect strength/intensity"
                }),
                "variation": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Variation/mode of the selected effect"
                }),
                "speed": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Animation speed factor"
                }),
                "feedback_blend": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.5,
                    "max": 0.999,
                    "step": 0.005,
                    "tooltip": "How much of previous frame to retain (higher = more trail effect)"
                }),
                "color_shift": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Amount of color transformation to apply"
                }),
                "reset": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset the feedback loop"
                }),
            }
        }
    
    def __init__(self):
        self.frame_count = 0
        self.time = 0
        self.last_image = None
        self.last_effect = None
        self.effect_buffers = {}
        
    def apply_effect(self, image, effect, intensity, variation, speed, feedback_blend, color_shift, reset):
        batch_size, height, width, channels = image.shape
        device = image.device
        
        # Reset state if requested or if effect changes
        if reset or effect != self.last_effect:
            self.frame_count = 0
            self.time = 0
            self.effect_buffers = {}
            
        # Update time and frame counter
        self.frame_count += 1
        frame_time = self.frame_count * speed * 0.03
        self.time = frame_time
        self.last_effect = effect
        
        # Fast path for batch processing - we'll process each image separately
        result_images = []
        
        for b in range(batch_size):
            img = image[b]
            result = self._process_single_image(
                img, effect, intensity, variation, 
                feedback_blend, color_shift, device, 
                width, height, frame_time, speed
            )
            result_images.append(result)

        return (torch.stack(result_images),)
    
    def _process_single_image(self, img, effect, intensity, variation, 
                             feedback_blend, color_shift, device, 
                             width, height, time, speed):
        # Highly optimized processing path - calculate grid transformations in one go
        # Coordinates grid (only calculate once and reuse)
        if not hasattr(self, 'grid_cache') or self.grid_cache['size'] != (height, width):
            y_norm, x_norm = torch.meshgrid(
                torch.linspace(-1, 1, height, device=device),
                torch.linspace(-1, 1, width, device=device),
                indexing='ij'
            )
            self.grid_cache = {
                'size': (height, width),
                'x_norm': x_norm, 
                'y_norm': y_norm,
                'r': torch.sqrt(x_norm**2 + y_norm**2),
                'theta': torch.atan2(y_norm, x_norm)
            }
        
        # Get cached grids
        x_norm = self.grid_cache['x_norm'] 
        y_norm = self.grid_cache['y_norm']
        r = self.grid_cache['r']
        theta = self.grid_cache['theta']
        
        # Initialize source coordinates to identity
        source_x = x_norm.clone()
        source_y = y_norm.clone()

        # Apply selected effect transformation
        if effect == "fractal_zoom":
            # Fractal zoom with organic movement
            zoom = 0.98 + math.sin(time * 2.5) * 0.02
            
            # Variation affects spiral pattern
            spiral = variation * 0.2 * intensity
            angle = theta + r * spiral + time * speed
            
            # Complex transform that creates evolving spiral patterns
            dx = torch.sin(r * (3 + variation) + time * 3) * 0.05 * intensity
            dy = torch.cos(r * (3 + variation) + time * 3.5) * 0.05 * intensity
            
            # Apply zoom
            source_x = (x_norm * zoom + dx) * (1 + 0.03 * intensity * math.sin(time))
            source_y = (y_norm * zoom + dy) * (1 + 0.03 * intensity * math.cos(time))
            
            # Add swirling effect
            twist = r * (4.0 * intensity) + time * (1.5 + variation * 0.3)
            source_x += torch.sin(twist + theta) * 0.05 * intensity
            source_y += torch.cos(twist - theta) * 0.05 * intensity
            
        elif effect == "kaleidoscope":
            # Kaleidoscope effect with dynamic symmetry
            segments = 4 + variation * 2
            segment_angle = 2 * torch.pi / segments
            
            # Rotate over time
            rot_theta = theta + time * (0.2 + 0.1 * variation) 
            
            # Mirror segments
            segment_id = torch.floor(rot_theta / segment_angle)
            in_segment = rot_theta - segment_id * segment_angle
            flip_mask = (segment_id % 2 == 0)
            mirror_theta = torch.where(flip_mask, in_segment, segment_angle - in_segment)
            
            # Apply transform with zoom breathing and ripples
            zoom = 0.8 + 0.2 * math.sin(time * 1.5) + r * 0.1 * intensity
            ripple = torch.sin(r * (8 + variation) - time * 3) * 0.06 * intensity
            
            source_x = torch.cos(mirror_theta) * r * (zoom + ripple)
            source_y = torch.sin(mirror_theta) * r * (zoom + ripple)
        
        elif effect == "plasma_field":
            # Plasma with flow dynamics
            scale = 5.0 + variation * 2.0
            speed_factor = 3.0 + variation * 0.5
            
            # Create plasma-like distortion fields
            plasma_x = torch.sin(x_norm * scale + time * speed_factor) * torch.cos(y_norm * (scale*0.8) + time)
            plasma_y = torch.sin(y_norm * scale - time * (speed_factor*0.7)) * torch.cos(x_norm * (scale*0.9) - time * 1.3)
            plasma_xy = torch.sin((x_norm + y_norm) * (scale*0.4) + time * 1.5) * 0.3
            
            # Combine fields for rich movement
            flow_x = (plasma_x + plasma_xy) * 0.15 * intensity
            flow_y = (plasma_y + plasma_xy) * 0.15 * intensity
            
            # Apply field distortion
            source_x = x_norm + flow_x
            source_y = y_norm + flow_y
        
        elif effect == "quantum_wave":
            # Quantum wave effect with interference patterns
            time_factor = time * (2.0 + variation * 0.4)
            
            # Dynamic wave parameters
            wave_scale = (4.0 + variation) + math.sin(time) * 1.5
            wave_speed = 1.5 + variation * 0.2
            
            # Create interference pattern
            wave1 = torch.sin(x_norm * wave_scale + time_factor * wave_speed)
            wave2 = torch.sin(y_norm * wave_scale - time_factor * wave_speed)
            wave3 = torch.sin((x_norm + y_norm) * wave_scale * 0.5 + time_factor * 1.3)
            wave4 = torch.sin(r * wave_scale * 0.8 - time_factor)
            
            # Combine wave patterns
            interference = (wave1 + wave2 + wave3 + wave4) * 0.25
            
            # Apply distortion with intensity scaling
            dist_scale = 0.12 * intensity * (1.0 + 0.3 * math.sin(time * 0.5))
            source_x = x_norm + interference * dist_scale * (1.0 + 0.5 * x_norm)
            source_y = y_norm + interference * dist_scale * (1.0 + 0.5 * y_norm)
            
            # Add subtle spiral
            rotation = time * 0.3 * (1 + variation * 0.1)
            rot_x = x_norm * torch.cos(torch.tensor(rotation, device=device)) - y_norm * torch.sin(torch.tensor(rotation, device=device))
            rot_y = x_norm * torch.sin(torch.tensor(rotation, device=device)) + y_norm * torch.cos(torch.tensor(rotation, device=device))
            
            # Blend rotation into final coordinates
            source_x = source_x * 0.85 + rot_x * 0.15
            source_y = source_y * 0.85 + rot_y * 0.15
        
        elif effect == "neural_dream":
            # Neural network-like patterns that evolve
            # Create layered activation patterns
            scale_factor = 3.0 + variation * 0.5
            time_offset = time * (1.2 + variation * 0.2)
            
            # First activation layer - positional encodings
            act1 = torch.sin(x_norm * scale_factor + time_offset)
            act2 = torch.sin(y_norm * scale_factor * 1.1 - time_offset * 0.7)
            act3 = torch.sin((x_norm*0.5 + y_norm*1.5) * scale_factor * 0.7 + time_offset * 0.5)
            
            # Second layer - non-linear combinations
            layer2_1 = torch.tanh(act1 * act2 * 0.5 + act3 * 0.3)
            layer2_2 = torch.tanh(act2 * act3 * 0.5 + act1 * 0.3)
            
            # Output layer with residual connection
            dx = layer2_1 * 0.15 * intensity
            dy = layer2_2 * 0.15 * intensity
            
            # Apply distortion with subtle spiral
            angle = r * (variation * 0.2) + time * (0.2 + variation * 0.05)
            source_x = x_norm + dx + torch.sin(angle) * r * 0.02 * intensity
            source_y = y_norm + dy + torch.cos(angle) * r * 0.02 * intensity
        
        elif effect == "digital_flow":
            # Digital fluid-like motion with emergent patterns
            # Base flow field
            scale = 4.0 + variation
            time_factor = time * (1.0 + variation * 0.2)
            
            # Create flow vectors using simplex-like patterns
            flow_x1 = torch.sin(x_norm * scale + time_factor) * torch.cos(y_norm * (scale*0.5) + time_factor * 0.6)
            flow_y1 = torch.cos(x_norm * (scale*0.6) - time_factor * 0.7) * torch.sin(y_norm * scale - time_factor)
            
            # Second flow field at different scale
            flow_x2 = torch.sin((x_norm + y_norm) * (scale*0.4) + time_factor * 1.3) 
            flow_y2 = torch.cos((y_norm - x_norm) * (scale*0.4) - time_factor * 1.1)
            
            # Combine flow fields
            flow_x = (flow_x1 * 0.6 + flow_x2 * 0.4) * 0.12 * intensity
            flow_y = (flow_y1 * 0.6 + flow_y2 * 0.4) * 0.12 * intensity
            
            # Apply flow with subtle zoom pulse
            zoom = 1.0 + math.sin(time * 1.5) * 0.01 * intensity
            source_x = x_norm * zoom + flow_x
            source_y = y_norm * zoom + flow_y
            
            # Add digital quantization if higher variation
            if variation >= 3:
                # Quantize stronger for higher variations
                quant_level = 15 + (5 - variation) * 5  # More divisions for lower variation
                source_x = torch.round(source_x * quant_level) / quant_level
                source_y = torch.round(source_y * quant_level) / quant_level
        
        elif effect == "hypershock":
            # High energy effect with shockwaves and energy fields
            # Base frequencies
            freq = 5.0 + variation
            time_factor = time * (2.0 + variation * 0.3)
            
            # Create pulsing shockwaves
            shock1_radius = (time_factor * 0.7) % 2.0 - 1.0
            shock1 = torch.exp(-((r - shock1_radius) * 10)**2) * 0.5
            
            shock2_radius = (time_factor * 0.5 + 1.0) % 2.0 - 1.0
            shock2 = torch.exp(-((r - shock2_radius) * 10)**2) * 0.4
            
            # Energy field
            energy_x = torch.sin(x_norm * freq + time_factor * 1.5) * 0.1
            energy_y = torch.sin(y_norm * freq - time_factor * 1.3) * 0.1
            
            # Combine effects
            dx = (shock1 + shock2 * 0.7) * x_norm * 0.3 + energy_x
            dy = (shock1 + shock2 * 0.7) * y_norm * 0.3 + energy_y
            
            # Apply distortion with high energy scaling
            source_x = x_norm + dx * intensity
            source_y = y_norm + dy * intensity
            
            # Add spiral warp for higher variations
            if variation >= 2:
                warp = torch.sin(r * (8 - variation) + time_factor) * 0.1 * intensity
                source_x += torch.sin(theta + torch.tensor(time, device=device)) * warp
                source_y += torch.cos(theta + torch.tensor(time, device=device)) * warp
        
        # Ensure coordinates stay within bounds
        source_x = torch.clamp(source_x, -1.98, 1.98)
        source_y = torch.clamp(source_y, -1.98, 1.98)
        
        # Convert from normalized coords to actual pixel sampling grid [-1, 1]
        grid = torch.stack([source_x, source_y], dim=-1)
        grid = grid.unsqueeze(0)  # Add batch dimension
        
        # Sample the image (fast tensor operation)
        img_rgb = img[..., :3].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        sampled = torch.nn.functional.grid_sample(
            img_rgb, grid, 
            mode='bilinear', 
            padding_mode='reflection',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # Back to [H, W, C]
        
        # Apply color transformations based on selected effect
        if color_shift > 0:
            # Optimize color processing by pushing calculations to the GPU
            if effect == "fractal_zoom":
                # Vibrant color amplification
                color_factor = time * 0.1 % 1.0
                hsv_rotation = torch.tensor([
                    [math.cos(color_factor * 6.28), -math.sin(color_factor * 6.28), 0],
                    [math.sin(color_factor * 6.28), math.cos(color_factor * 6.28), 0],
                    [0, 0, 1]
                ], device=device)
                
                flat_rgb = sampled.reshape(-1, 3)
                rotated_rgb = torch.matmul(flat_rgb, hsv_rotation)
                sampled = rotated_rgb.reshape(sampled.shape)
                
                # Vibrance enhancement
                luma = sampled[..., 0] * 0.299 + sampled[..., 1] * 0.587 + sampled[..., 2] * 0.114
                chroma = sampled - luma.unsqueeze(-1)
                sampled = luma.unsqueeze(-1) + chroma * (1 + color_shift * 0.5)
                
            elif effect in ["kaleidoscope", "hypershock"]:
                # High contrast with glow
                # Add bloom/glow to bright areas
                bloom_threshold = 0.7
                bloom_mask = (sampled.mean(dim=-1) > bloom_threshold).float().unsqueeze(-1)
                bloom = sampled * bloom_mask * color_shift * 0.5
                
                # Apply bloom with additive blending
                sampled = torch.clamp(sampled + bloom, 0, 1)
                
                # Edge enhancement for kaleidoscope
                if effect == "kaleidoscope":
                    edge_x = torch.abs(sampled[:, 1:] - sampled[:, :-1]).mean(-1, keepdim=True)
                    edge_x = torch.cat([edge_x, edge_x[:, -1:]], dim=1)
                    
                    edge_y = torch.abs(sampled[1:, :] - sampled[:-1, :]).mean(-1, keepdim=True)
                    edge_y = torch.cat([edge_y, edge_y[-1:, :]], dim=0)
                    
                    edge = (edge_x + edge_y) * 0.5
                    sampled = torch.clamp(sampled + edge * color_shift, 0, 1)
            
            elif effect in ["plasma_field", "neural_dream"]:
                # Hue rotation with time
                hue_shift = (time * 0.05) % 1.0
                
                # RGB to HLS-like transform (simplified)
                cmax, _ = torch.max(sampled, dim=-1, keepdim=True)
                cmin, _ = torch.min(sampled, dim=-1, keepdim=True)
                delta = cmax - cmin
                
                # Calculate luminance
                luma = (cmax + cmin) * 0.5
                
                # Calculate saturation
                sat = torch.where(delta > 0, delta / (1 - torch.abs(2 * luma - 1) + 1e-6), torch.zeros_like(delta))
                
                # Enhance saturation
                sat = torch.clamp(sat * (1 + color_shift), 0, 1)
                
                # Apply new saturation while preserving luminance
                normalized = (sampled - cmin) / (delta + 1e-6)
                shifted = luma + (normalized - 0.5) * sat * (1 - torch.abs(2 * luma - 1))
                sampled = torch.clamp(shifted, 0, 1)
                
            elif effect == "quantum_wave":
                # Quantum interference colors
                # Shift RGB channels with interference pattern
                shift = torch.sin(r * 4 + time * 2) * 0.1 * color_shift
                
                # RGB channel shifting
                r_shift = torch.roll(sampled[..., 0], shifts=int(shift[0,0].item() * width), dims=0)
                r_shift = torch.roll(r_shift, shifts=int(shift[0,0].item() * height), dims=1)
                
                b_shift = torch.roll(sampled[..., 2], shifts=-int(shift[0,0].item() * width), dims=0)
                b_shift = torch.roll(b_shift, shifts=-int(shift[0,0].item() * height), dims=1)
                
                sampled = torch.stack([r_shift, sampled[..., 1], b_shift], dim=-1)
                
            elif effect == "digital_flow":
                # Digital color quantization
                if color_shift > 0.3:
                    quant_levels = max(3, int(10 * (1 - color_shift)))  # Fewer levels with higher color_shift
                    sampled = torch.floor(sampled * quant_levels) / quant_levels
        
        # Alpha handling
        if img.shape[-1] == 4:
            # Keep original alpha channel
            alpha = torch.nn.functional.grid_sample(
                img[..., 3:].permute(2, 0, 1).unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='reflection',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            
            # Recombine with RGB
            sampled = torch.cat([sampled, alpha], dim=-1)
        
        # Apply feedback blend if we have a last image
        if self.last_image is not None and img.shape == self.last_image.shape:
            # Optimize blend operation
            result = sampled * (1.0 - feedback_blend) + self.last_image * feedback_blend
        else:
            result = sampled
        
        # Store for next frame
        self.last_image = result
        
        return result
    
class ToString:
    """A node that converts any input to a string."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"),{}),  # Accept any input type
                "format_string": ("STRING", {
                    "default": "{}",
                    "multiline": False,
                    "tooltip": "Optional format string (e.g., '{:.2f}' for 2 decimal place float)"
                })
            },
            "optional": {
                "prepend": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text to add before the converted string"
                }),
                "append": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text to add after the converted string"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"
    CATEGORY = "utils"
    
    def to_string(self, any, format_string="{}", prepend="", append=""):
        try:
            # Try to use the format string if provided
            result = format_string.format(any)
        except (ValueError, TypeError):
            # Fall back to simple string conversion if format fails
            result = str(any)
        
        # Add prepend and append text
        result = prepend + result + append
            
        return (result,)

class RoundNode:
    """A node that rounds a float to a specified number of decimal places."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Float value to round"
                }),
                "decimal_places": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of decimal places to round to"
                })
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("rounded_value",)
    FUNCTION = "round_value"
    CATEGORY = "utils"
    
    def round_value(self, value, decimal_places):
        if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            rounded = [round(v, decimal_places) for v in value]
        else:
            rounded = round(value, decimal_places)
        return (rounded,)
    

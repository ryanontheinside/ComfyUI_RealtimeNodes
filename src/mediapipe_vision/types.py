from typing import List, Optional, NamedTuple, Tuple
import numpy as np # Added numpy for matrix type

# Define a structure for a single landmark point
class LandmarkPoint(NamedTuple):
    index: int
    x: float
    y: float
    z: float
    visibility: Optional[float] = None # Available in Pose landmarks
    presence: Optional[float] = None   # Available in Pose landmarks

# Define container structures for the output of each landmark task
# These will be returned by the respective detector nodes

class FaceLandmarksResult(NamedTuple):
    landmarks: List[LandmarkPoint]
    # Add other face-specific results if needed later (e.g., bounding box)

class HandLandmarksResult(NamedTuple):
    landmarks: List[LandmarkPoint]
    world_landmarks: Optional[List[LandmarkPoint]] # Keep world landmarks optional for now
    handedness: Optional[str] # 'Left', 'Right'

class PoseLandmarksResult(NamedTuple):
    landmarks: List[LandmarkPoint]
    world_landmarks: Optional[List[LandmarkPoint]]
    # Segmentation mask info could potentially go here if tightly coupled

# We might add more types later for detections, segmentations etc.
# For now, focus is on landmarks.

# Define the list types that will be used for batching
# These will be the types used in node RETURN_TYPES/INPUT_TYPES strings
FACE_LANDMARKS_LIST = List[List[FaceLandmarksResult]] # Batch[FacesPerImage[Result]]
HAND_LANDMARKS_LIST = List[List[HandLandmarksResult]] # Batch[HandsPerImage[Result]]
POSE_LANDMARKS_LIST = List[List[PoseLandmarksResult]] # Batch[PosesPerImage[Result]]

# --- Blendshape Result --- 
class Blendshape(NamedTuple):
    index: Optional[int]
    score: float
    category_name: str # e.g., mouthSmileLeft, eyeBlinkRight
    display_name: Optional[str]

# --- Common Types --- 
class BoundingBox(NamedTuple):
    origin_x: int
    origin_y: int
    width: int
    height: int

# --- Face Detection Result --- 
class FaceKeypoint(NamedTuple):
    label: Optional[str] # e.g., 'nose_tip', 'left_eye'
    x: float
    y: float

class FaceDetectionResult(NamedTuple):
    bounding_box: BoundingBox
    keypoints: Optional[List[FaceKeypoint]]
    score: Optional[float] # Confidence score for the detection

# --- Object Detection Result --- 
class ObjectDetectionCategory(NamedTuple):
    index: Optional[int]
    score: float
    display_name: Optional[str]
    category_name: Optional[str]

class ObjectDetectionResult(NamedTuple):
    bounding_box: BoundingBox
    categories: List[ObjectDetectionCategory]

# --- Batch List Types for Nodes --- 
# Define the list types that will be used for batching
# These will be the types used in node RETURN_TYPES/INPUT_TYPES strings
FACE_LANDMARKS = List[List[FaceLandmarksResult]] # Batch[FacesPerImage[Result]]
HAND_LANDMARKS = List[List[HandLandmarksResult]] # Batch[HandsPerImage[Result]]
POSE_LANDMARKS = List[List[PoseLandmarksResult]] # Batch[PosesPerImage[Result]]
OBJECT_DETECTIONS = List[List[ObjectDetectionResult]] # Batch[DetectionsPerImage[Result]]
FACE_DETECTIONS = List[List[FaceDetectionResult]] # Batch[DetectionsPerImage[Result]]

# --- Gesture Recognition Result --- 
class GestureCategory(NamedTuple):
    index: Optional[int]
    score: float
    display_name: Optional[str]
    category_name: Optional[str]

class GestureRecognitionResult(NamedTuple):
    gestures: List[GestureCategory] # Top gesture(s) recognized for a hand
    handedness: Optional[str] # Which hand this gesture belongs to

# Define Batch List Type
GESTURE_RECOGNITIONS = List[List[GestureRecognitionResult]] # Batch[Hands[Gestures]]

# --- Image Embedding Result ---
class EmbeddingEntry(NamedTuple):
    index: int
    score: float
    display_name: Optional[str]
    category_name: Optional[str]

class ImageEmbedderResult(NamedTuple):
    float_embedding: Optional[List[float]]
    quantized_embedding: Optional[bytes]
    head_index: int
    head_name: Optional[str]

# Define Batch List Type
IMAGE_EMBEDDINGS = List[List[ImageEmbedderResult]] # Batch[EmbeddingsPerHead[Result]]

# --- Interactive Segmentation Input/Result ---
class PointOfInterest(NamedTuple):
    x: float
    y: float
    label: Optional[str] = None # Optional label for the point

# Result is typically a mask, potentially multiple if categories are involved
# Using existing MASK type might suffice, or define a specific one if needed.

# --- Holistic Landmark Result --- 
class HolisticLandmarksResult(NamedTuple):
    face_landmarks: Optional[List[LandmarkPoint]] # Only one face per person
    pose_landmarks: Optional[List[LandmarkPoint]]
    pose_world_landmarks: Optional[List[LandmarkPoint]]
    left_hand_landmarks: Optional[List[LandmarkPoint]]
    right_hand_landmarks: Optional[List[LandmarkPoint]]
    # Pose segmentation mask could also be included here

# Define Batch List Type
HOLISTIC_LANDMARKS = List[HolisticLandmarksResult] # Batch[ResultPerPerson] 

# =======================================
# --- Batch List Types for Node I/O --- 
# =======================================
FACE_LANDMARKS = List[List[FaceLandmarksResult]] # Batch[FacesPerImage[Result]]
HAND_LANDMARKS = List[List[HandLandmarksResult]] # Batch[HandsPerImage[Result]]
POSE_LANDMARKS = List[List[PoseLandmarksResult]] # Batch[PosesPerImage[Result]]
OBJECT_DETECTIONS = List[List[ObjectDetectionResult]] # Batch[DetectionsPerImage[Result]]
FACE_DETECTIONS = List[List[FaceDetectionResult]] # Batch[DetectionsPerImage[Result]]
GESTURE_RECOGNITIONS = List[List[GestureRecognitionResult]] # Batch[Hands[Gestures]]
IMAGE_EMBEDDINGS = List[List[ImageEmbedderResult]] # Batch[EmbeddingsPerHead[Result]]
HOLISTIC_LANDMARKS = List[HolisticLandmarksResult] # Batch[ResultPerPerson] 

# New Types - Defined AFTER Blendshape class
# Updated names to match FaceLandmarker output convention
BLENDSHAPES_LIST = List[List[List[Blendshape]]] # Batch[FacesPerImage[BlendshapeList[Blendshape]]]
TRANSFORM_MATRIX_LIST = List[List[np.ndarray]] # Batch[FacesPerImage[Matrix]] 
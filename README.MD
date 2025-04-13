#staged  for integration


## ✨ Overview

This repository aims to provide a complete and centralized implementation of all MediaPipe vision tasks, optimized for real-time use in ComfyUI.

These tools can be used for interactive AI art, responsive interfaces, motion tracking, advanced masking workflows, and more. These are well optimizede for real-time usage (with comfystream), but are blazing fast for normal batch processing as well.

## 🚀 Features

| Category | Available Tools |
|----------|-------------|
| **Face Analysis** | Face detection, face mesh (478 points), blendshapes, head pose |
| **Body Tracking** | Pose estimation (33 landmarks), segmentation masks |
| **Hand Analysis** | Hand tracking (21 landmarks per hand), gesture recognition |
| **Image Processing** | Object detection, image segmentation, image embeddings |
| **Creative Tools** | Face stylization, interactive segmentation |
| **Control Nodes** | Use deltas from tracking landmarks to control other Comfy nodes |

## 📋 Supported MediaPipe Tasks

* **Face Detection:** Face bounding boxes and keypoints
* **Face Landmark Detection:** Face mesh landmarks with optional expression analysis
* **Hand Landmark Detection:** Hand position tracking with 21 landmarks
* **Pose Landmark Detection:** Body pose tracking with 33 landmarks
* **Object Detection:** Common object detection using models like EfficientDet
* **Image Segmentation:** Category-based image segmentation
* **Gesture Recognition:** Recognition of common hand gestures
* **Image Embedding:** Feature vector generation for image similarity
* **Interactive Segmentation:** User-guided image masking
* **Face Stylization:** Artistic style application to faces
* **Holistic Landmark Detection:** Full-body landmark detection (legacy)

> **Note:** Holistic landmark detection is currently using legacy API as we await official Tasks API release.



## 🧩 Components

The extension organizes MediaPipe functionality into these components:

1. **Model Loaders** - Each task has a loader node that prepares the appropriate model
2. **Processing Nodes** - Task-specific nodes that process images
3. **Visualization Nodes** - Optional nodes to render detection results
4. **Control Nodes** - Delta tracking and movement detection to control parameters
5. **Utility Nodes** - Helper nodes for masks, tracking, and result processing

## ⚙️ Control Nodes

Control nodes convert MediaPipe landmark tracking into ComfyUI parameters, enabling dynamic control of your workflows:

### Types of Control Nodes

- **Delta Control Nodes** - Convert movement of tracked points into INT/FLOAT values
- **Head Pose Control Nodes** - Convert face orientation (yaw/pitch/roll) into parameters
- **Trigger Nodes** - Create boolean triggers when movements or poses cross thresholds

### Available Controls:

| Feature | Control Types | Example Applications |
|---------|--------------|---------------------|
| **Face** | Head Pose (yaw/pitch/roll/position), Blendshape expressions | Camera control, gaze-directed generation, emotion-based parameters |
| **Hand** | Landmark delta tracking, finger position/movement | UI control, gesture-based adjustments, pinch-to-zoom effects |
| **Pose** | Body landmark movement, joint tracking | Animation control, posture-based parameters |

### Head Pose Controls

The Head Pose Control nodes provide these specific controls:
- **Yaw** - Head turning left/right (horizontal rotation)
- **Pitch** - Head nodding up/down (vertical rotation)
- **Roll** - Head tilting side to side
- **X/Y/Z Position** - 3D spatial position of head

### Facial Expression Controls

MediaPipe's face landmark detection includes blendshape coefficients that can be used to control parameters based on facial expressions. There are ~40 expression attributes that can be used. Each expression can be mapped to INT/FLOAT outputs for precise control over generation parameters, or used as triggers for workflow events.

### Usage Examples

- Connect index finger movement to control seed randomization
- Use head yaw position to control left/right panning in a scene
- Trigger workflow steps when a user nods (pitch movement)
- Map hand pinching gesture to control zoom level
- Adjust diffusion strength based on how fast someone moves
- Use smile intensity to control positive prompt weighting
- Trigger different styles when specific facial expressions are detected

Check out the [examples directory](examples/) for sample workflows demonstrating how to use the control nodes with different MediaPipe features.


## 🛠️ Installation

Use the ComfyUI-Manager, or....

```bash
# Navigate to your ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes

# Clone the repository
git clone https://github.com/your-username/ComfyUI-MediaPipe-Vision.git

# Enter the directory
cd ComfyUI-MediaPipe-Vision

# Install dependencies
pip install -r requirements.txt

# Restart ComfyUI
```

> **Note:** GPU Support varies by platform. Generally, for Linux platforms, you can reference [these instructions](https://ai.google.dev/edge/mediapipe/framework/getting_started/gpu_support) to enable GPU support.

## 🔍 Usage Guide

### Basic Workflow

1. Add a `Load ... Model (MediaPipe)` node for your task
2. Connect your image and the model info to the processor node
3. Use the outputs with visualization nodes or in your ComfyUI workflow

### Example: Face Landmark Detection

```
[Load Face Landmarker Model] → model_info → [Face Landmarker] ← image
                                               |
                                               ↓ landmarks
                                 [Visualize Face Landmarks] ← original_image
                                               |
                                               ↓ visualization
                                           [Preview]
```

### Example: Movement Tracking

```
[Load Hand Landmarker Model] → model_info → [Hand Landmarker] ← webcam_image
                                               |
                                               ↓ landmarks
                         [Hand Landmark Delta Float Control] (index_finger_tip)
                                               |
                                               ↓ float_value
                                      [Any Comfy Parameter]
```

## 📓 Technical Notes

* Models download automatically to `ComfyUI/models/mediapipe/<task_type>/`
* Nodes support batch processing for multiple images
* CPU processing works on most systems; GPU support depends on configuration

## 🤝 Contributing

Contributions are welcome! For bugs or suggestions for improvements, please open an issue or submit a pull request.

**Feature Requests Strongly Encouraged!** This project provides a flexible infrastructure that can be adapted to many different use cases. While several basic capabilities are implemented, the project aims to address more use cases and problems:

- Have an idea for creative AI interactions using vision?
- Need a specific type of landmark tracking or detection?
- Working on a unique workflow that could benefit from real-time vision?
- Found limitations in the current implementation for your use case?

Please open an issue to discuss your needs even if you're not sure how to implement them. The MediaPipe framework is powerful and extensible, and this project aims to make that power accessible within ComfyUI for any computer vision application.

## 📜 License

[MIT License](LICENSE)




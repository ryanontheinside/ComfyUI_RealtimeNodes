import torch
import numpy as np
import mediapipe as mp
import cv2

class HandTrackingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_confidence": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.1, 
                    "max": 1.0,
                    "step": 0.05
                }),
                "tracking_confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "max_num_hands": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "static_image_mode": ("BOOLEAN", {
                    "default": False
                }),
                "draw_debug": ("BOOLEAN", {
                    "default": False
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "HAND_DATA")
    RETURN_NAMES = ("debug_image", "hand_data")
    FUNCTION = "track_hands"
    CATEGORY = "image/processing"
    DESCRIPTION = "(((EXPERIMENTAL))) Track hands in an image using MediaPipe. Returns a debug image with landmarks and a dictionary of hand data."
    def __init__(self):
        self.last_detection_conf = 0.5
        self.last_tracking_conf = 0.5
        self.last_max_hands = 2
        self.last_static_mode = True
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.last_static_mode,
            max_num_hands=self.last_max_hands,
            min_detection_confidence=self.last_detection_conf,
            min_tracking_confidence=self.last_tracking_conf
        )

    def track_hands(self, image, detection_confidence, tracking_confidence, max_num_hands=2, static_image_mode=True, draw_debug=True):
        # Check if we need to reinitialize MediaPipe hands
        if (detection_confidence != self.last_detection_conf or 
            tracking_confidence != self.last_tracking_conf or 
            max_num_hands != self.last_max_hands or
            static_image_mode != self.last_static_mode):
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self.last_detection_conf = detection_confidence
            self.last_tracking_conf = tracking_confidence
            self.last_max_hands = max_num_hands
            self.last_static_mode = static_image_mode

        # Handle input
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        batch_size = image_np.shape[0]

        # Process each image in batch
        processed_images = []
        hand_data_batch = []
        
        for i in range(batch_size):
            single_image = image_np[i]
            image_rgb = cv2.cvtColor(single_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            hand_data = {
                "left_hand": None,
                "right_hand": None,
                "num_hands": 0,
                "hands_present": False
            }
            
            if results.multi_hand_landmarks:
                hand_data["num_hands"] = len(results.multi_hand_landmarks)
                hand_data["hands_present"] = True
                
                # Create single debug image for all hands
                debug_image = single_image.copy() if draw_debug else single_image
                
                # Process all detected hands
                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_type = handedness.classification[0].label.lower()
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    # Store hand data based on type
                    if hand_type == "left" and hand_data["left_hand"] is None:
                        hand_data["left_hand"] = landmarks
                    elif hand_type == "right" and hand_data["right_hand"] is None:
                        hand_data["right_hand"] = landmarks
                    
                    # Draw landmarks for this hand on the shared debug image
                    if draw_debug:
                        mp.solutions.drawing_utils.draw_landmarks(
                            debug_image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                
                processed_images.append(debug_image)
            else:
                processed_images.append(single_image)
            
            hand_data_batch.append(hand_data)

        # Convert back to tensors
        processed_batch = np.stack(processed_images, axis=0)
        output_tensor = torch.from_numpy(processed_batch).float() / 255.0
        
        return (output_tensor, hand_data_batch)

    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()

class HandMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hand_data": ("HAND_DATA",),
                "mask_type": (["full_hand", "palm_only", "fingers_only", "thumb", "index", "middle", "ring", "pinky"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "create_mask"
    CATEGORY = "image/processing"

    def create_mask(self, image, hand_data, mask_type):
        batch_size, height, width, channels = image.shape
        masks = np.zeros((batch_size, height, width), dtype=np.float32)
        
        # Finger indices in MediaPipe hand landmarks
        finger_indices = {
            "thumb": [1,2,3,4],
            "index": [5,6,7,8],
            "middle": [9,10,11,12],
            "ring": [13,14,15,16],
            "pinky": [17,18,19,20]
        }
        palm_indices = [0,1,5,9,13,17]  # Wrist and base of each finger
        
        for i in range(batch_size):
            frame_hand_data = hand_data[i]
            if not frame_hand_data["hands_present"]:
                continue
                
            frame_mask = np.zeros((height, width), dtype=np.float32)
            
            for hand_key in ["right_hand", "left_hand"]:
                hand = frame_hand_data[hand_key]
                if hand is None:
                    continue
                    
                points = hand[:, :2]  # Take only x,y
                points = (points * np.array([width, height])).astype(np.int32)
                
                if mask_type == "full_hand":
                    hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(frame_mask, hull, 1.0)
                
                elif mask_type == "palm_only":
                    palm_points = points[palm_indices]
                    hull = cv2.convexHull(palm_points)
                    cv2.fillConvexPoly(frame_mask, hull, 1.0)
                
                elif mask_type == "fingers_only":
                    for finger in finger_indices.values():
                        finger_points = points[finger]
                        cv2.fillConvexPoly(frame_mask, cv2.convexHull(finger_points), 1.0)
                
                elif mask_type in finger_indices:
                    finger_points = points[finger_indices[mask_type]]
                    cv2.fillConvexPoly(frame_mask, cv2.convexHull(finger_points), 1.0)
            
            masks[i] = frame_mask
            
            # # Debug output
            # if frame_hand_data["right_hand"] is not None:
            #     print(f"Right hand points range: x({np.min(points[:, 0]):.1f}-{np.max(points[:, 0]):.1f}), y({np.min(points[:, 1]):.1f}-{np.max(points[:, 1]):.1f})")
            # if frame_hand_data["left_hand"] is not None:
            #     print(f"Left hand points range: x({np.min(points[:, 0]):.1f}-{np.max(points[:, 0]):.1f}), y({np.min(points[:, 1]):.1f}-{np.max(points[:, 1]):.1f})")
        
        mask_tensor = torch.from_numpy(masks)
        return (image, mask_tensor) 
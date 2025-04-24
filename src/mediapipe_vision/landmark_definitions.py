"""
MediaPipe landmark definitions for the different landmark types.
These are used to provide more descriptive tooltips in the landmark position nodes.
"""

# Face landmarks (468 points)
# Key landmarks with their descriptions
FACE_LANDMARK_DESCRIPTIONS = {
    0: "Chin bottom",
    1: "Chin left-of-center",
    4: "Left chin corner",
    5: "Left jaw angle",
    7: "Left ear tragion",
    8: "Left ear lobe",
    10: "Right chin corner",
    13: "Right ear tragion",
    14: "Right ear lobe",
    17: "Left eyebrow outer corner",
    21: "Left eyebrow inner corner",
    22: "Right eyebrow inner corner",
    26: "Right eyebrow outer corner",
    33: "Left eye outer corner",
    40: "Left eye inner corner",
    39: "Left eye bottom",
    37: "Left eye top",
    42: "Right eye inner corner",
    45: "Right eye bottom",
    43: "Right eye top",
    46: "Right eye outer corner",
    51: "Nose bridge top",
    4: "Nose bridge bottom",
    57: "Nose tip",
    62: "Nose left nostril corner",
    66: "Nose right nostril corner",
    76: "Left side of nose",
    82: "Right side of nose",
    61: "Left nostril",
    67: "Right nostril",
    84: "Left corner of mouth",
    78: "Upper lip top center",
    80: "Upper lip bottom center",
    90: "Right corner of mouth",
    87: "Bottom lip top center",
    86: "Bottom lip bottom center",
    14: "Upper teeth",
    13: "Lower teeth",
    
    # General regions (for tooltip summary)
    # 0-17: Jaw/face outline
    # 18-35: Eyebrows
    # 36-71: Eyes
    # 48-67: Nose
    # 68-83: Lips outer contour
    # 84-95: Lips inner contour
    # 96-145: Face contour
    # 150-168: Left eye iris (if using refined landmarks)
    # 468-477: Refined eye landmarks (if enabled)
}

# Tooltip summarizing face landmarks
FACE_LANDMARK_TOOLTIP = """MediaPipe Face Landmarks (0-467):
0-17: Face outline/jaw
18-35: Eyebrows
36-71: Eyes
48-67: Nose
68-95: Lips
96-145: Face contour
Notable: 
  0: Chin bottom
  4/10: Chin corners
  57: Nose tip
  84/90: Mouth corners
  33/46: Eye outer corners"""

# Hand landmarks (21 points)
HAND_LANDMARK_DESCRIPTIONS = {
    0: "Wrist",
    1: "Thumb CMC", 
    2: "Thumb MCP",
    3: "Thumb IP",
    4: "Thumb tip",
    5: "Index finger MCP",
    6: "Index finger PIP",
    7: "Index finger DIP",
    8: "Index finger tip",
    9: "Middle finger MCP",
    10: "Middle finger PIP",
    11: "Middle finger DIP",
    12: "Middle finger tip",
    13: "Ring finger MCP",
    14: "Ring finger PIP",
    15: "Ring finger DIP",
    16: "Ring finger tip",
    17: "Pinky MCP",
    18: "Pinky PIP",
    19: "Pinky DIP",
    20: "Pinky tip"
}

# Tooltip for hand landmarks
HAND_LANDMARK_TOOLTIP = """MediaPipe Hand Landmarks (0-20):
0: Wrist
1-4: Thumb (1:base to 4:tip)
5-8: Index finger (5:base to 8:tip)
9-12: Middle finger (9:base to 12:tip)
13-16: Ring finger (13:base to 16:tip)
17-20: Pinky finger (17:base to 20:tip)
MCP: Metacarpophalangeal joint (knuckles)
PIP: Proximal interphalangeal joint (middle joint)
DIP: Distal interphalangeal joint (joint near fingertip)"""

# Pose landmarks (33 points)
POSE_LANDMARK_DESCRIPTIONS = {
    0: "Nose",
    1: "Left eye (inner)",
    2: "Left eye",
    3: "Left eye (outer)",
    4: "Right eye (inner)",
    5: "Right eye",
    6: "Right eye (outer)",
    7: "Left ear",
    8: "Right ear",
    9: "Mouth (left)",
    10: "Mouth (right)",
    11: "Left shoulder",
    12: "Right shoulder",
    13: "Left elbow",
    14: "Right elbow",
    15: "Left wrist",
    16: "Right wrist",
    17: "Left pinky knuckle",
    18: "Right pinky knuckle",
    19: "Left index knuckle",
    20: "Right index knuckle",
    21: "Left thumb tip",
    22: "Right thumb tip",
    23: "Left hip",
    24: "Right hip",
    25: "Left knee",
    26: "Right knee",
    27: "Left ankle",
    28: "Right ankle",
    29: "Left heel",
    30: "Right heel",
    31: "Left foot index",
    32: "Right foot index"
}

# Tooltip for pose landmarks
POSE_LANDMARK_TOOLTIP = """MediaPipe Pose Landmarks (0-32):
0-10: Face (0:nose, 1-6:eyes, 7-8:ears, 9-10:mouth)
11-22: Upper body
  11-12: Shoulders
  13-14: Elbows
  15-16: Wrists
  17-22: Hands
23-32: Lower body
  23-24: Hips
  25-26: Knees
  27-28: Ankles
  29-32: Feet""" 
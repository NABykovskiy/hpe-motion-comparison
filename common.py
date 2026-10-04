"""Общие константы эксперимента: канонический формат COCO-17 и группы частей тела."""

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Индексы точек COCO-17 по функциональным группам (для вектора v_t)
GROUPS = {
    "head": [0, 1, 2, 3, 4],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
}


def keypoints_columns():
    """Схема единого CSV: frame, time_s, затем x/y/c для каждой из 17 точек."""
    cols = ["frame", "time_s"]
    for i in range(17):
        cols += [f"x{i}", f"y{i}", f"c{i}"]
    return cols


# Маппинг "нативный формат модели -> COCO-17": для каждого COCO-индекса
# указан индекс точки в исходном формате модели.

# BlazePose (33 точки, MediaPipe Tasks PoseLandmarker)
BLAZEPOSE_TO_COCO17 = [
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]

# OpenPose BODY_25 (25 точек, порядок из официальной документации OpenPose):
# 0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist, 5 LShoulder, 6 LElbow,
# 7 LWrist, 8 MidHip, 9 RHip, 10 RKnee, 11 RAnkle, 12 LHip, 13 LKnee,
# 14 LAnkle, 15 REye, 16 LEye, 17 REar, 18 LEar, 19-24 ступни.
BODY25_TO_COCO17 = [
    0,   # nose
    16,  # left_eye
    15,  # right_eye
    18,  # left_ear
    17,  # right_ear
    5,   # left_shoulder
    2,   # right_shoulder
    6,   # left_elbow
    3,   # right_elbow
    7,   # left_wrist
    4,   # right_wrist
    12,  # left_hip
    9,   # right_hip
    13,  # left_knee
    10,  # right_knee
    14,  # left_ankle
    11,  # right_ankle
]

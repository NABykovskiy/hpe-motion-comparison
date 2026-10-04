"""Диагностика: работает ли GPU-делегат MediaPipe Tasks на этой машине."""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = "models/pose_landmarker_full.task"


def try_delegate(delegate_name):
    delegate = (
        mp_python.BaseOptions.Delegate.GPU
        if delegate_name == "GPU"
        else mp_python.BaseOptions.Delegate.CPU
    )
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL, delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        lm.detect_for_video(mp_img, 0)


print(f"mediapipe version: {getattr(mp, '__version__', '?')}")

for name in ("CPU", "GPU"):
    try:
        try_delegate(name)
        print(f"[{name}] OK")
    except Exception as e:
        print(f"[{name}] FAIL: {type(e).__name__}: {e}")

"""BlazePose: видео -> единый формат результатов эксперимента.

Реализация: MediaPipe Tasks API (PoseLandmarker), режим VIDEO.
Выход:
  results/blazepose/keypoints.csv  — frame, time_s, x/y/c для 17 точек COCO (пиксели);
                                      кадр без детекции — строка из NaN
  results/blazepose/timing.json    — параметры прогона и метрики производительности

Время обработки измеряется на ВТОРОМ проходе (первый — прогрев модели);
в замер входит декодирование кадров, чтобы метрика была сопоставима между моделями.
"""

import argparse
import datetime as dt
import json
import os
import platform
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from common import BLAZEPOSE_TO_COCO17, keypoints_columns


def cpu_name():
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0]
    except OSError:
        return platform.processor()

COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def run_pass(video_path, landmarker, fps, collect, ts_offset_ms=0):
    """Один проход по видео. collect=True — сохранять ключевые точки.

    ts_offset_ms — смещение таймстампов для второго прохода: MediaPipe требует
    монотонно возрастающие таймстампы в течение жизни одного инстанса модели.
    """
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        base_ms = int(round(frame_idx * 1000.0 / fps)) if fps > 0 else frame_idx
        ts_ms = ts_offset_ms + base_ms
        result = landmarker.detect_for_video(mp_image, ts_ms)
        if collect:
            kps = [np.nan] * 51
            if result.pose_landmarks:
                # num_poses=1: первый (и единственный) человек
                lms = result.pose_landmarks[0]
                for coco_i, blaze_i in enumerate(BLAZEPOSE_TO_COCO17):
                    lm = lms[blaze_i]
                    kps[coco_i * 3 + 0] = lm.x * width
                    kps[coco_i * 3 + 1] = lm.y * height
                    kps[coco_i * 3 + 2] = lm.visibility
            time_s = frame_idx / fps if fps > 0 else np.nan
            rows.append([frame_idx, time_s] + kps)
        frame_idx += 1
    cap.release()
    return rows, frame_idx, (width, height)


def save_preview(video_path, df, out_path, fps):
    """Аннотированное видео для визуальной проверки (не влияет на тайминг)."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height)
    )
    for _, row in df.iterrows():
        ok, frame = cap.read()
        if not ok:
            break
        pts = {}
        for i in range(17):
            x, y, c = row[f"x{i}"], row[f"y{i}"], row[f"c{i}"]
            if pd.notna(x) and pd.notna(c) and c > 0.5:
                pts[i] = (int(x), int(y))
        for a, b in COCO17_EDGES:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 200, 0), 1)
        for p in pts.values():
            cv2.circle(frame, p, 3, (0, 255, 0), -1)
        writer.write(frame)
    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="data/test_dance_30s.mp4")
    ap.add_argument("--model", default="models/pose_landmarker_full.task")
    ap.add_argument("--out", default="results/blazepose")
    ap.add_argument("--num-poses", type=int, default=1)
    ap.add_argument("--preview", action="store_true",
                    help="сохранить аннотированное видео preview.mp4")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Видео не найдено: {video_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    base = mp_tasks.BaseOptions(model_asset_path=str(args.model))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_vision.PoseLandmarker.create_from_options(opts) as landmarker:
        # Проход 1: прогрев (результаты не сохраняются)
        _, n_warmup, size = run_pass(video_path, landmarker, fps, collect=False)
        # Проход 2: измеряемый (таймстампы смещены за пределы первого прохода)
        t0 = time.perf_counter()
        rows, n_frames, size = run_pass(
            video_path, landmarker, fps, collect=True,
            ts_offset_ms=(n_warmup + 1) * 1000,
        )
        t_proc = time.perf_counter() - t0

    df = pd.DataFrame(rows, columns=keypoints_columns())
    df.to_csv(out_dir / "keypoints.csv", index=False)

    conf_cols = [f"c{i}" for i in range(17)]
    n_detected = int(df[conf_cols].notna().any(axis=1).sum())
    mean_conf = float(df[conf_cols].stack().mean())

    timing = {
        "model": "BlazePose",
        "implementation": f"MediaPipe {getattr(mp, '__version__', '?')} Tasks API (PoseLandmarker)",
        "model_file": Path(args.model).name,
        "num_poses": args.num_poses,
        "thresholds": {
            "min_pose_detection_confidence": 0.5,
            "min_pose_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "device": "CPU",
        "cpu": {
            "model": cpu_name(),
            "logical_processors": os.cpu_count(),
        },
        "video": {
            "file": video_path.name,
            "resolution": f"{size[0]}x{size[1]}",
            "fps": round(fps, 3) if fps else None,
        },
        "frames_processed": n_frames,
        "frames_with_detection": n_detected,
        "detection_rate": round(n_detected / n_frames, 4) if n_frames else None,
        "mean_confidence": round(mean_conf, 4) if not np.isnan(mean_conf) else None,
        "processing_time_s": round(t_proc, 3),
        "processing_fps": round(n_frames / t_proc, 2) if t_proc > 0 else None,
        "timing_note": "второй проход (после прогревочного), включая декодирование кадров",
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.preview:
        save_preview(video_path, df, out_dir / "preview.mp4", fps)

    print(f"Кадров обработано:      {n_frames} (прогрев: {n_warmup})")
    print(f"Кадров с детекцией:     {n_detected} ({timing['detection_rate']:.1%})")
    print(f"Средний confidence:     {timing['mean_confidence']}")
    print(f"Время обработки:        {t_proc:.2f} с -> {timing['processing_fps']} FPS")
    print(f"Результаты:             {out_dir}/keypoints.csv, timing.json")
    if args.preview:
        print(f"Превью:                 {out_dir}/preview.mp4")


if __name__ == "__main__":
    main()

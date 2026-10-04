"""RTMO: видео -> единый формат результатов эксперимента.

Реализация: rtmlib 0.0.16, одностадийная модель RTMO-m (детекция + поза
в одной сети, вход 640x640), инференс через ONNX Runtime на CPU.
Выход:
  results/rtmo/keypoints.csv  — frame, time_s, x/y/c для 17 точек COCO (пиксели);
                                 кадр без детекции — строка из NaN
  results/rtmo/timing.json    — параметры прогона и метрики производительности

Время обработки измеряется на ВТОРОМ проходе (первый — прогрев модели);
в замер входит декодирование кадров, чтобы метрика была сопоставима между моделями.
"""

import argparse
import datetime as dt
import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rtmlib import RTMO

from common import keypoints_columns

MODEL = "models/rtmo_m_body7_640x640.onnx"


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


def run_pass(video_path, rtmo, fps, collect):
    """Один проход по видео. collect=True — сохранять ключевые точки.

    При отсутствии детекций RTMO возвращает фоллбэк-позу из нулей со скорами 0
    (не пустой массив) — такой кадр записывается как NaN-строка.
    При нескольких детекциях берётся поза с индексом 0: NMS в rtmlib
    возвращает людей по убыванию score.
    """
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kps = [np.nan] * 51
        keypoints, scores = rtmo(frame)
        if keypoints.shape[0] > 0 and float(scores[0].max()) > 0.0:
            kpt, scr = keypoints[0], scores[0]
            for i in range(17):
                kps[i * 3 + 0] = float(kpt[i, 0])
                kps[i * 3 + 1] = float(kpt[i, 1])
                kps[i * 3 + 2] = float(scr[i])
        if collect:
            time_s = frame_idx / fps if fps > 0 else np.nan
            rows.append([frame_idx, time_s] + kps)
        frame_idx += 1
    cap.release()
    return rows, frame_idx


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
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="results/rtmo")
    ap.add_argument("--preview", action="store_true",
                    help="сохранить аннотированное видео preview.mp4")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Видео не найдено: {video_path}")
    if not Path(args.model).exists():
        raise SystemExit(f"Модель не найдена: {args.model}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    # RTMO инстанцируется напрямую, а не через Body: Body для rtmo-моделей
    # перезаписывает локальный путь на URL из своего каталога (баг rtmlib 0.0.16)
    rtmo = RTMO(
        onnx_model=str(args.model),
        model_input_size=(640, 640),
        to_openpose=False,  # порядок точек COCO-17 (как в остальных моделях)
        backend="onnxruntime",
        device="cpu",
    )

    # Проход 1: прогрев (результаты не сохраняются)
    _, n_warmup = run_pass(video_path, rtmo, fps, collect=False)
    # Проход 2: измеряемый
    t0 = time.perf_counter()
    rows, n_frames = run_pass(video_path, rtmo, fps, collect=True)
    t_proc = time.perf_counter() - t0

    df = pd.DataFrame(rows, columns=keypoints_columns())
    df.to_csv(out_dir / "keypoints.csv", index=False)

    conf_cols = [f"c{i}" for i in range(17)]
    n_detected = int(df[conf_cols].notna().any(axis=1).sum())
    mean_conf = float(df[conf_cols].stack().mean())

    try:
        rtmlib_ver = pkg_version("rtmlib")
    except PackageNotFoundError:
        rtmlib_ver = "?"

    timing = {
        "model": "RTMO",
        "implementation": (
            f"rtmlib {rtmlib_ver} one-stage: RTMO-m (детекция+поза, 640x640), "
            f"ONNX Runtime CPU"
        ),
        "model_files": [Path(args.model).name],
        "person_selection": "поза с максимальным score (top-1 по NMS)",
        "thresholds": {
            "score_thr": 0.7,
            "nms_thr": 0.45,
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

"""OpenPose: видео -> единый формат результатов эксперимента.

Реализация: официальный portable-билд OpenPose 1.7.0 (CMU) для Windows,
CPU-версия. Модель BODY_25 (bottom-up, Part Affinity Fields), вызов — внешний
процесс bin/OpenPoseDemo.exe с JSON-выводом (--write_json), без рендеринга
и без окна отображения.
Выход:
  results/openpose/keypoints.csv   — frame, time_s, x/y/c для 17 точек COCO (пиксели);
                                     кадр без детекции — строка из NaN
  results/openpose/timing.json     — параметры прогона и метрики производительности
  results/openpose/openpose_json/  — сырые JSON OpenPose (BODY_25) для трассировки

Время обработки измеряется на ВТОРОМ вызове OpenPoseDemo по полному видео.
Прогрев — короткий вызов на первых кадрах того же видео: OpenPose — внешний
процесс, каждый вызов заново загружает модель, и полный прогревочный проход
ничего не даёт сверх прогрева кэша ОС (веса, видео, DLL).
"""

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import BODY25_TO_COCO17, keypoints_columns

OPENPOSE_DIR = Path(__file__).resolve().parent / "openpose"


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


def openpose_version(openpose_root):
    vt = openpose_root / "Version.txt"
    m = None
    try:
        m = re.search(r"(\d+\.\d+\.\d+)", vt.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        pass
    return m.group(1) if m else "1.7.0"


COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def run_openpose_demo(exe, openpose_root, video_path, json_dir, frame_last=None):
    """Один вызов OpenPoseDemo.exe: обработка видео, JSON на каждый кадр.

    frame_last: обработать только кадры 0..frame_last включительно (для
    прогрева и тестовых прогонов).
    --number_people_max 1: протокол эксперимента — один человек.
    --render_pose 0 и --display 0: без визуализации (не тратим CPU на отрисовку).
    Вывод OpenPoseDemo идёт прямо в консоль — виден прогресс долгого прогона.
    """
    json_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--video", str(video_path),
        "--model_pose", "BODY_25",
        "--model_folder", str(openpose_root / "models"),
        "--number_people_max", "1",
        "--write_json", str(json_dir),
        "--render_pose", "0",
        "--display", "0",
    ]
    if frame_last is not None:
        cmd += ["--frame_first", "0", "--frame_last", str(frame_last)]
    result = subprocess.run(cmd, cwd=str(openpose_root))
    if result.returncode != 0:
        raise SystemExit(
            f"OpenPoseDemo.exe завершился с кодом {result.returncode}"
        )


def count_frames_from_json(json_dir):
    """Число обработанных кадров = максимальный индекс JSON-файла + 1.

    OpenPose именует файлы '<имя видео>_<12-значный индекс>_keypoints.json'.
    """
    indices = []
    for p in json_dir.glob("*_keypoints.json"):
        m = re.search(r"_(\d{12})_keypoints\.json$", p.name)
        if m:
            indices.append(int(m.group(1)))
    return (max(indices) + 1) if indices else 0


def parse_json_dir(json_dir, n_frames, fps, video_stem):
    """JSON OpenPose -> строки единого CSV (BODY_25 -> COCO-17).

    Кадр без детекции — файл с пустым списком people (или отсутствующий файл)
    — записывается как NaN-строка.
    """
    rows = []
    n_missing_files = 0
    for frame_idx in range(n_frames):
        kps = [np.nan] * 51
        path = json_dir / f"{video_stem}_{frame_idx:012d}_keypoints.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if people:
                body25 = people[0].get("pose_keypoints_2d", [])
                if len(body25) >= 75:
                    for coco_i, src_i in enumerate(BODY25_TO_COCO17):
                        kps[coco_i * 3 + 0] = float(body25[src_i * 3 + 0])
                        kps[coco_i * 3 + 1] = float(body25[src_i * 3 + 1])
                        kps[coco_i * 3 + 2] = float(body25[src_i * 3 + 2])
        else:
            n_missing_files += 1
        time_s = frame_idx / fps if fps > 0 else np.nan
        rows.append([frame_idx, time_s] + kps)
    if n_missing_files:
        print(f"Предупреждение: не хватает {n_missing_files} JSON-файлов")
    return rows


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
    ap.add_argument("--openpose", default=str(OPENPOSE_DIR),
                    help="корень распакованного OpenPose (с bin/, models/)")
    ap.add_argument("--out", default="results/openpose")
    ap.add_argument("--warmup-frames", type=int, default=30,
                    help="сколько кадров гнать в прогревочном вызове")
    ap.add_argument("--limit", type=int, default=None,
                    help="обработать только первые N кадров (для тестовых прогонов)")
    ap.add_argument("--preview", action="store_true",
                    help="сохранить аннотированное видео preview.mp4")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise SystemExit(f"Видео не найдено: {video_path}")
    openpose_root = Path(args.openpose).resolve()
    exe = openpose_root / "bin" / "OpenPoseDemo.exe"
    if not exe.exists():
        raise SystemExit(f"OpenPoseDemo.exe не найден: {exe}")
    model_weights = (
        openpose_root / "models" / "pose" / "body_25" / "pose_iter_584000.caffemodel"
    )
    if not model_weights.exists():
        raise SystemExit(f"Веса BODY_25 не найдены: {model_weights}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    warmup_dir = out_dir / "_warmup_json"
    json_dir = out_dir / "openpose_json"
    last_frame = (args.limit - 1) if args.limit else None

    print(f"Прогрев ({args.warmup_frames} кадров)...")
    if warmup_dir.exists():
        shutil.rmtree(warmup_dir)
    t0 = time.perf_counter()
    run_openpose_demo(exe, openpose_root, video_path, warmup_dir,
                      frame_last=args.warmup_frames - 1)
    t_warmup = time.perf_counter() - t0
    shutil.rmtree(warmup_dir)

    print("Измеряемый проход (полное видео)...")
    if json_dir.exists():
        shutil.rmtree(json_dir)
    t0 = time.perf_counter()
    run_openpose_demo(exe, openpose_root, video_path, json_dir,
                      frame_last=last_frame)
    t_proc = time.perf_counter() - t0

    n_frames = count_frames_from_json(json_dir)
    if n_frames == 0:
        raise SystemExit("OpenPose не записал ни одного JSON-файла")

    rows = parse_json_dir(json_dir, n_frames, fps, video_path.stem)
    df = pd.DataFrame(rows, columns=keypoints_columns())
    df.to_csv(out_dir / "keypoints.csv", index=False)

    conf_cols = [f"c{i}" for i in range(17)]
    n_detected = int(df[conf_cols].notna().any(axis=1).sum())
    mean_conf = float(df[conf_cols].stack().mean())

    version = openpose_version(openpose_root)
    timing = {
        "model": "OpenPose",
        "implementation": (
            f"OpenPose {version} (официальный Windows CPU-билд CMU), "
            f"BODY_25 (bottom-up, Part Affinity Fields), Caffe CPU"
        ),
        "model_files": [model_weights.name],
        "person_selection": "number_people_max=1 (OpenPose отбирает одного человека)",
        "parameters": {
            "model_pose": "BODY_25",
            "net_resolution": "default (-1x368)",
            "render_pose": 0,
            "display": 0,
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
        "timing_note": (
            "второй вызов OpenPoseDemo по полному видео (после короткого "
            "прогревочного вызова на первых кадрах того же видео, см. поле "
            "warmup), включая декодирование видео и однократную загрузку "
            "модели в начале вызова (у Python-моделей загрузка выполнена до "
            "замера). Полный прогревочный проход не выполнялся: OpenPose — "
            "внешний процесс, каждый вызов перезагружает модель, между "
            "вызовами переносится только кэш ОС"
        ),
        "warmup": {
            "frames": args.warmup_frames,
            "time_s": round(t_warmup, 3),
        },
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.preview:
        save_preview(video_path, df, out_dir / "preview.mp4", fps)

    print(f"Кадров обработано:      {n_frames}")
    print(f"Кадров с детекцией:     {n_detected} ({timing['detection_rate']:.1%})")
    print(f"Средний confidence:     {timing['mean_confidence']}")
    print(f"Время обработки:        {t_proc:.2f} с -> {timing['processing_fps']} FPS")
    print(f"Результаты:             {out_dir}/keypoints.csv, timing.json")
    print(f"Сырые JSON:             {json_dir}")
    if args.preview:
        print(f"Превью:                 {out_dir}/preview.mp4")


if __name__ == "__main__":
    main()

"""Расчёт временных признаков движения v_t из keypoints.csv (модель-независимый).

Реализует раздел 2.3 статьи:
  1) нормализация координат: опорная точка mid-hip, масштаб s_t = длина торса;
  2) перемещение точки d_k^t между кадрами t-1 и t по нормализованным координатам;
  3) m_j^t = среднее d_k^t по группе G_j (голова, л/п рука, л/п нога);
  4) v_t = [m_head, m_larm, m_rarm, m_lleg, m_rleg].

Учёт пропусков: точка считается определённой, если её confidence >= CONF_THR
в обоих кадрах пары; m_j = NaN, если в группе нет ни одной валидной точки.

Критерий временной устойчивости (раздел 2.4):
  D_j = среднее |m_j^t - m_j^{t-1}| по валидным парам кадров.

Выход:
  results/<model>/motion_features.csv — frame, time_s, m_head, ..., m_rleg
  results/<model>/summary.json        — агрегаты по группам
  results/<model>/motion_timeseries.png — график пяти компонент v_t
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import GROUPS

CONF_THR = 0.3          # порог достоверности ключевой точки
MIN_SCALE_PX = 10.0     # защита от деления на ~0 при потере торса

# COCO-17: 5,6 — плечи; 11,12 — бёдра
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    n = len(df)
    xy = np.full((n, 17, 2), np.nan)
    conf = np.full((n, 17), np.nan)
    for i in range(17):
        xy[:, i, 0] = df[f"x{i}"]
        xy[:, i, 1] = df[f"y{i}"]
        conf[:, i] = df[f"c{i}"]
    valid = conf >= CONF_THR
    return df, xy, valid


def normalize(xy, valid):
    """Координаты относительно mid-hip, масштаб — длина торса (torso units)."""
    n = xy.shape[0]
    ref = np.full((n, 2), np.nan)
    scale = np.full(n, np.nan)
    both_hips = valid[:, L_HIP] & valid[:, R_HIP]
    ref[both_hips] = (xy[both_hips, L_HIP] + xy[both_hips, R_HIP]) / 2.0
    both_sho = valid[:, L_SHO] & valid[:, R_SHO]
    mid_sho = np.full((n, 2), np.nan)
    mid_sho[both_sho] = (xy[both_sho, L_SHO] + xy[both_sho, R_SHO]) / 2.0
    ok = both_hips & both_sho
    scale[ok] = np.linalg.norm(mid_sho[ok] - ref[ok], axis=1)
    scale[scale < MIN_SCALE_PX] = np.nan
    xyn = (xy - ref[:, None, :]) / scale[:, None, None]
    return xyn


def compute_motion(xyn, valid):
    """d_k^t (кадр t против t-1) и m_j^t по пяти группам."""
    n = xyn.shape[0]
    d = np.full((n, 17), np.nan)  # d[t] — перемещение между t-1 и t; d[0] = NaN
    for t in range(1, n):
        both = valid[t - 1] & valid[t]
        diff = xyn[t] - xyn[t - 1]
        dist = np.linalg.norm(diff, axis=1)
        d[t] = np.where(both, dist, np.nan)

    m = {name: np.full(n, np.nan) for name in GROUPS}
    for name, idxs in GROUPS.items():
        for t in range(1, n):
            vals = [d[t, k] for k in idxs if not np.isnan(d[t, k])]
            if vals:
                m[name][t] = float(np.mean(vals))
    return d, m


def temporal_instability(series):
    """D_j = среднее |m_t - m_{t-1}| по валидным парам (NaN разрывает пару)."""
    diffs = []
    for t in range(1, len(series)):
        a, b = series[t - 1], series[t]
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(abs(b - a))
    return float(np.mean(diffs)) if diffs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="папка results/<model> с keypoints.csv")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    df, xy, valid = load_keypoints(model_dir / "keypoints.csv")
    xyn = normalize(xy, valid)
    d, m = compute_motion(xyn, valid)

    out = pd.DataFrame({
        "frame": df["frame"],
        "time_s": df["time_s"],
        "m_head": m["head"],
        "m_left_arm": m["left_arm"],
        "m_right_arm": m["right_arm"],
        "m_left_leg": m["left_leg"],
        "m_right_leg": m["right_leg"],
    })
    out.to_csv(model_dir / "motion_features.csv", index=False)

    n = len(out)
    timing = json.loads((model_dir / "timing.json").read_text(encoding="utf-8"))
    summary = {
        "model": timing.get("model"),
        "video": timing.get("video", {}).get("file"),
        "frames": n,
        "conf_threshold": CONF_THR,
        "groups": {},
    }
    for name, col in [("head", "m_head"), ("left_arm", "m_left_arm"),
                      ("right_arm", "m_right_arm"), ("left_leg", "m_left_leg"),
                      ("right_leg", "m_right_leg")]:
        s = out[col].values
        summary["groups"][name] = {
            "mean_m": round(float(np.nanmean(s)), 4) if not np.all(np.isnan(s)) else None,
            "median_m": round(float(np.nanmedian(s)), 4) if not np.all(np.isnan(s)) else None,
            "D_j": round(temporal_instability(s), 4),
            "valid_frames": int(np.sum(~np.isnan(s))),
            "valid_rate": round(float(np.sum(~np.isnan(s))) / n, 4),
        }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # График пяти компонент v_t
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    labels = [("m_head", "голова"), ("m_left_arm", "левая рука"),
              ("m_right_arm", "правая рука"), ("m_left_leg", "левая нога"),
              ("m_right_leg", "правая нога")]
    for ax, (col, title) in zip(axes, labels):
        ax.plot(out["time_s"], out[col], lw=0.8, color="tab:blue")
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title(
        f"Компоненты вектора v_t — {timing.get('model')} "
        f"(видео: {timing.get('video', {}).get('file')})", fontsize=11
    )
    axes[-1].set_xlabel("время, с")
    fig.tight_layout()
    fig.savefig(model_dir / "motion_timeseries.png", dpi=150)

    print(f"Модель: {summary['model']}  |  кадров: {n}")
    print(f"{'группа':<12}{'mean m':>10}{'D_j':>10}{'валидных':>12}")
    for name in summary["groups"]:
        g = summary["groups"][name]
        print(f"{name:<12}{g['mean_m']:>10.4f}{g['D_j']:>10.4f}"
              f"{g['valid_rate']:>11.0%}")
    print(f"Файлы: motion_features.csv, summary.json, motion_timeseries.png в {model_dir}")


if __name__ == "__main__":
    main()

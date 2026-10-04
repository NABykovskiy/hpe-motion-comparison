"""Точечная статистика для главы 4: conf по ключевым точкам, пропуски, выбросы,
тяжёлый хвост m_j, диагностика джиттера нормализации.

Читает results/{model}/keypoints.csv и motion_features.csv.
Выводит отчёт в консоль (redirect -> results/point_stats_report.txt).

Терминология (глава 4):
- detection: человек найден в кадре (строка keypoints.csv не целиком NaN);
- conf < CONF_THR: точка ниже порога валидности (не участвует в m_j);
- NaN: точка отсутствует в выводе модели (OpenPose может не выдавать точку).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
MODELS = ["blazepose", "rtmpose", "rtmo", "openpose"]
CONF_THR = 0.3
FRAME_W, FRAME_H = 1920, 1080

POINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank",
]


def load(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    kp = pd.read_csv(RESULTS / model / "keypoints.csv")
    mf = pd.read_csv(RESULTS / model / "motion_features.csv")
    return kp, mf


def point_report(model: str, kp: pd.DataFrame) -> None:
    ccols = [f"c{i}" for i in range(17)]
    det = ~kp[ccols].isna().all(axis=1)
    sub = kp[det]
    print(f"== {model} == детекция {len(sub)}/{len(kp)}")
    low_tot = nan_tot = tot = 0
    for i in range(17):
        c = sub[f"c{i}"].to_numpy(dtype=float)
        nan = int(np.isnan(c).sum())
        low = int(((c < CONF_THR) & ~np.isnan(c)).sum())
        med = np.nanmedian(c)
        tot += len(c); low_tot += low; nan_tot += nan
        print(f"  {i:2d} {POINT_NAMES[i]:<6s} med={med:.3f} low={low:4d} nan={nan:4d}")
    print(f"  ИТОГО точек={tot} low={low_tot} ({100*low_tot/tot:.2f}%) "
          f"nan={nan_tot} ({100*nan_tot/tot:.2f}%)")
    xb = yb = 0
    for i in range(17):
        x = kp[f"x{i}"].to_numpy()
        y = kp[f"y{i}"].to_numpy()
        xb += int(((x < 0) | (x > FRAME_W)).sum())
        yb += int(((y < 0) | (y > FRAME_H)).sum())
    print(f"  вне границ кадра: x={xb} y={yb} из {tot}")


def tail_report(model: str, mf: pd.DataFrame) -> None:
    v = mf["m_head"].dropna().to_numpy()
    k = int(np.ceil(0.05 * len(v)))
    s = np.sort(v)[::-1]
    share = s[:k].sum() / s.sum()
    print(f"{model:<11} n={len(v):3d} top-5% кадров дают {share*100:.0f}% суммы m_head, "
          f"mean={v.mean():.4f} median={np.median(v):.4f}")


def rtmpose_jitter(kp: pd.DataFrame, mf: pd.DataFrame) -> None:
    print("\n== Диагностика всплесков m_head (RTMPose): сырые опорные величины ==")
    xhip = (kp["x11"].to_numpy() + kp["x12"].to_numpy()) / 2
    yhip = (kp["y11"].to_numpy() + kp["y12"].to_numpy()) / 2
    xsho = (kp["x5"].to_numpy() + kp["x6"].to_numpy()) / 2
    ysho = (kp["y5"].to_numpy() + kp["y6"].to_numpy()) / 2
    hip = np.stack([xhip, yhip], 1)
    sho = np.stack([xsho, ysho], 1)
    d_hip = np.linalg.norm(np.diff(hip, axis=0), axis=1)
    d_sho = np.linalg.norm(np.diff(sho, axis=0), axis=1)
    torso = np.linalg.norm(sho - hip, axis=1)
    chin = (kp["c11"].to_numpy() + kp["c12"].to_numpy()) / 2
    top = mf.dropna(subset=["m_head"]).nlargest(10, "m_head")
    print(f"медиана d(hip)={np.nanmedian(d_hip):.1f}px, d(sho)={np.nanmedian(d_sho):.1f}px, "
          f"торс={np.nanmedian(torso):.1f}px")
    print(" кадр   m_head   d(hip)px  d(sho)px  торс(t-1)->t  conf(hip)")
    for f in top["frame"]:
        t = int(f)
        dh = d_hip[t - 1] if t >= 1 else np.nan
        ds = d_sho[t - 1] if t >= 1 else np.nan
        print(f"{t:5d} {mf['m_head'][t]:.4f}  {dh:8.1f}  {ds:8.1f}  "
              f"{torso[t-1]:5.1f}->{torso[t]:5.1f}   {chin[t]:.3f}")


def main() -> None:
    for m in MODELS:
        kp, mf = load(m)
        point_report(m, kp)
        print()
    print("=== Тяжёлый хвост m_head ===")
    for m in MODELS:
        _, mf = load(m)
        tail_report(m, mf)
    kp, mf = load("rtmpose")
    rtmpose_jitter(kp, mf)


if __name__ == "__main__":
    main()

"""Диагностика выбросов m_head RTMPose: источники скачков, s(t), разложение по точкам, V_common.

Покрывает шаги 1-4 плана рецензии:
 1) топ-20 кадров по m_head RTMPose: сырые скачки 5 точек головы (px),
    скачки mid-hip / mid-shoulder (px), скачки нормализационного масштаба s(t) (px);
 2) дамп сырых координат t-2..t+2 для топ-5 выбросов;
 3) графики: results/cross_scale_timeseries.png (s(t), 4 модели, отметки выбросов),
    results/rtmpose/rtmpose_head_decomposition.png (m_head = среднее 5 точек);
 4) проверка на общем подмножестве V_common: попадание топ-выбросов,
    сравнение RTMPose mean/median/D_j/автокорреляция: полный ряд vs V_common.

Вывод в консоль (redirect -> results/rtmpose_spike_report.txt).
Дополнительно: results/rtmpose/rtmpose_frame_diagnostics.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_motion import (
    CONF_THR, MIN_SCALE_PX, L_SHO, R_SHO, L_HIP, R_HIP,
    load_keypoints, normalize, compute_motion,
)
from cross_model_analysis import load_series, mean_abs_diff, autocorr_strict
from cross_timeseries_plot import COLORS, LABELS, PARTERRE
from common import COCO17_NAMES

MODELS = ["blazepose", "rtmpose", "rtmo", "openpose"]
HEAD = list(range(5))  # nose, l_eye, r_eye, l_ear, r_ear
DUMP_POINTS = [0, 1, 2, 3, 4, 5, 6, 11, 12]  # голова + плечи + бёдра
LOC_JUMP_PX = 200.0    # порог «грубой» ошибки локализации точки
SCALE_JUMP = 2.0       # отношение s(t)/s(t-1), считающееся скачком масштаба


def scale_and_ref(xy: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Опорная точка mid-hip, нормализационный масштаб s(t) и mid-shoulder — как в analyze_motion.normalize."""
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
    return ref, scale, mid_sho


def raw_pair_dist(xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Сырые перемещения точек между t-1 и t в px; NaN, если точка невалидна в одном из кадров."""
    n = xy.shape[0]
    out = np.full((n, xy.shape[1]), np.nan)
    for t in range(1, n):
        both = valid[t - 1] & valid[t]
        dist = np.linalg.norm(xy[t] - xy[t - 1], axis=1)
        out[t] = np.where(both, dist, np.nan)
    return out


def pair_step(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Перемещение агрегата (mid-hip / mid-shoulder) между t-1 и t в px."""
    n = len(mask)
    out = np.full(n, np.nan)
    for t in range(1, n):
        if mask[t - 1] and mask[t]:
            out[t] = np.linalg.norm(points[t] - points[t - 1])
    return out


def main() -> None:
    results = Path(__file__).parent / "results"
    pd.set_option("display.width", 240)

    # --- загрузка RTMPose ---
    kp, xy, valid = load_keypoints(results / "rtmpose" / "keypoints.csv")
    n = len(kp)
    conf = np.column_stack([kp[f"c{i}"].to_numpy(dtype=float) for i in range(17)])
    xyn = normalize(xy, valid)
    d_norm, m = compute_motion(xyn, valid)
    ref, scale, mid_sho = scale_and_ref(xy, valid)
    both_hips = valid[:, L_HIP] & valid[:, R_HIP]
    both_sho = valid[:, L_SHO] & valid[:, R_SHO]

    frames = kp["frame"].to_numpy()
    time_s = kp["time_s"].to_numpy()
    d_raw = raw_pair_dist(xy, valid)
    d_midhip = pair_step(ref, both_hips)
    d_midsho = pair_step(mid_sho, both_sho)

    min_c_head = np.full(n, np.nan)
    for t in range(n):
        vals = conf[t, HEAD]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            min_c_head[t] = vals.min()

    # --- (4) общее подмножество V_common по m_head ---
    series = {mod: load_series(mod) for mod in MODELS}
    valid_all = pd.concat(
        {mod: series[mod]["m_head"].notna() for mod in MODELS}, axis=1
    ).all(axis=1)
    common_idx = valid_all.index[valid_all]
    common_set = set(common_idx)
    s_full = series["rtmpose"]["m_head"]
    s_common = s_full.reindex(common_idx)

    print("=== Диагностика выбросов m_head RTMPose ===")
    print(f"порог достоверности точки: {CONF_THR}; кадров: {n}; "
          f"V_common (m_head валиден у всех 4 моделей): {len(common_idx)}")

    # --- (1) таблица топ-20 выбросов ---
    score = np.where(np.isnan(m["head"]), -1.0, m["head"])
    top20 = np.argsort(score)[::-1][:20]
    print("\n--- Топ-20 кадров по m_head (RTMPose) ---")
    print("d_*(px) — сырое перемещение за 1 кадр; s — длина торса; "
          "loc_jump — скачок точки >200 px; scale_jump — s(t)/s(t-1) >2 или <0.5")
    rows = []
    for t in top20:
        head_ds = [d_raw[t, k] for k in HEAD]
        head_max = np.nanmax(head_ds) if not np.all(np.isnan(head_ds)) else np.nan
        ratio = (scale[t] / scale[t - 1]
                 if t >= 1 and not np.isnan(scale[t]) and not np.isnan(scale[t - 1]) else np.nan)
        rows.append({
            "frame": int(frames[t]), "time_s": round(float(time_s[t]), 2),
            "m_head": round(float(m["head"][t]), 3),
            "nose": round(head_ds[0], 0) if not np.isnan(head_ds[0]) else None,
            "l_eye": round(head_ds[1], 0) if not np.isnan(head_ds[1]) else None,
            "r_eye": round(head_ds[2], 0) if not np.isnan(head_ds[2]) else None,
            "l_ear": round(head_ds[3], 0) if not np.isnan(head_ds[3]) else None,
            "r_ear": round(head_ds[4], 0) if not np.isnan(head_ds[4]) else None,
            "max_head_px": round(head_max, 0) if not np.isnan(head_max) else None,
            "d_midhip": round(d_midhip[t], 0) if not np.isnan(d_midhip[t]) else None,
            "d_midsho": round(d_midsho[t], 0) if not np.isnan(d_midsho[t]) else None,
            "s_prev": round(scale[t - 1], 0) if t >= 1 and not np.isnan(scale[t - 1]) else None,
            "s_cur": round(scale[t], 0) if not np.isnan(scale[t]) else None,
            "s_ratio": round(ratio, 2) if not np.isnan(ratio) else None,
            "min_c_head": round(min_c_head[t], 3) if not np.isnan(min_c_head[t]) else None,
            "loc_jump": bool(not np.isnan(head_max) and head_max > LOC_JUMP_PX),
            "scale_jump": bool(not np.isnan(ratio) and (ratio > SCALE_JUMP or ratio < 1 / SCALE_JUMP)),
            "in_Vcommon": bool(int(frames[t]) in common_set),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    n_loc = int(table["loc_jump"].sum())
    n_sc = int(table["scale_jump"].sum())
    n_both = int((table["loc_jump"] & table["scale_jump"]).sum())
    print(f"\nиз топ-20: грубая ошибка локализации — {n_loc}, скачок масштаба — {n_sc}, "
          f"оба фактора — {n_both}, попадают в V_common — {int(table['in_Vcommon'].sum())}")

    # --- (2) дамп сырых координат для топ-5 ---
    print("\n--- Дамп сырых координат t-2..t+2 (топ-5 выбросов) ---")
    print("    точки: " + ", ".join(COCO17_NAMES[k] for k in DUMP_POINTS))
    for t in top20[:5]:
        print(f"\n>> frame {int(frames[t])} (t={time_s[t]:.2f} с), "
              f"m_head={m['head'][t]:.3f}, s={scale[t]:.0f} px")
        for dt in (-2, -1, 0, 1, 2):
            u = t + dt
            if u < 0 or u >= n:
                continue
            marks = " <== выброс" if dt == 0 else ""
            parts = [f"{COCO17_NAMES[k]}=({xy[u, k, 0]:.0f},{xy[u, k, 1]:.0f},c={conf[u, k]:.2f})"
                     for k in DUMP_POINTS]
            print(f"  t{'+' if dt >= 0 else ''}{dt}: " + " ".join(parts) + marks)

    # --- (4) RTMPose: полный ряд vs V_common ---
    print("\n--- RTMPose m_head: полный ряд vs V_common ---")
    print(f"{'':<12}{'n':>6}{'mean':>9}{'median':>9}{'D_j':>9}{'autocorr':>10}")
    for name, s in (("полный ряд", s_full), ("V_common", s_common)):
        print(f"{name:<12}{s.notna().sum():>6}{s.mean():>9.4f}{s.median():>9.4f}"
              f"{mean_abs_diff(s):>9.4f}{autocorr_strict(s):>10.3f}")

    # --- (3a) график s(t) для четырёх моделей ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for mod in MODELS:
        kpm, xym, valm = load_keypoints(results / mod / "keypoints.csv")
        _, sm, _ = scale_and_ref(xym, valm)
        ax.plot(kpm["time_s"], sm, lw=1.0, color=COLORS[mod], label=LABELS[mod], alpha=0.9)
    for st in time_s[top20[:10]]:
        ax.axvline(st, color="red", lw=0.7, ls="--", alpha=0.6)
    ax.axvspan(*PARTERRE, color="gray", alpha=0.15)
    ax.set_title("Нормализационный масштаб s(t) — длина торса, px", fontsize=11)
    ax.set_xlabel("время, с")
    ax.set_ylabel("s(t), px")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.annotate("эпизод партерной работы", xy=(PARTERRE[0] + 0.1, 0.93),
                xycoords=("data", "axes fraction"), fontsize=9, color="dimgray")
    ax.annotate("красные штрихи — топ-10 выбросов RTMPose", xy=(0.01, 0.03),
                xycoords="axes fraction", fontsize=9, color="red")
    fig.tight_layout()
    out_scale = results / "cross_scale_timeseries.png"
    fig.savefig(out_scale, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_scale}")

    # --- (3b) разложение m_head на 5 точек ---
    spike_mask = np.where(np.isnan(m["head"]), False, m["head"] > 1.0)
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(time_s, m["head"], lw=1.0, color="tab:red")
    axes[0].set_ylabel("m_head (ср.)", fontsize=9)
    for k in HEAD:
        axes[k + 1].plot(time_s, d_norm[:, k], lw=0.8, color="tab:blue")
        axes[k + 1].set_ylabel(COCO17_NAMES[k], fontsize=9)
    for ax in axes:
        for st in time_s[spike_mask]:
            ax.axvline(st, color="red", lw=0.5, ls="--", alpha=0.5)
        ax.axvspan(*PARTERRE, color="gray", alpha=0.15)
        ax.grid(alpha=0.3)
    axes[0].set_title("RTMPose: разложение m_head на перемещения отдельных точек головы "
                      "(нормализованные единицы)", fontsize=11)
    axes[0].annotate("красные штрихи — кадры с m_head > 1.0", xy=(0.01, 0.85),
                     xycoords="axes fraction", fontsize=9, color="red")
    axes[-1].set_xlabel("время, с")
    fig.tight_layout()
    out_dec = results / "rtmpose" / "rtmpose_head_decomposition.png"
    fig.savefig(out_dec, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out_dec}")

    # --- (5) сводный CSV ---
    diag = pd.DataFrame({
        "frame": frames,
        "time_s": time_s,
        "x_c": ref[:, 0],
        "y_c": ref[:, 1],
        "s_px": scale,
        "m_head": m["head"],
        "m_left_arm": m["left_arm"],
        "m_right_arm": m["right_arm"],
        "m_left_leg": m["left_leg"],
        "m_right_leg": m["right_leg"],
        "n_valid_head": valid[:, HEAD].sum(axis=1),
        "min_c_head": min_c_head,
    })
    out_csv = results / "rtmpose" / "rtmpose_frame_diagnostics.csv"
    diag.to_csv(out_csv, index=False)
    print(f"Сохранено: {out_csv}")


if __name__ == "__main__":
    main()

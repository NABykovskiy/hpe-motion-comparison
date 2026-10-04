"""Кросс-модельный анализ: сводная таблица m_j/D_j + корреляционные матрицы рядов m_j.

Читает results/{model}/motion_features.csv для всех завершённых прогонов.
Выводит в консоль и сохраняет в results/cross_model_summary.json.

Методические оговорки (глава 4):
- mean m_j смещён селективностью детекции (модель, пропускающая сложные кадры,
  выглядит «спокойнее») -> приводим mean И median;
- conf между моделями не сравнивается (разная семантика);
- корреляции считаем и по сырым, и по сглаженным (rolling 5) рядам;
- D_j и автокорреляция считаются ТОЛЬКО по парам строго соседних кадров
  (разница номеров кадров = 1): пары через пропуски не «склеиваются», иначе
  разность накапливает перемещение за несколько кадров.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
MODELS = ["blazepose", "rtmpose", "rtmo", "openpose"]
GROUPS = ["m_head", "m_left_arm", "m_right_arm", "m_left_leg", "m_right_leg"]


def load_series(model: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / model / "motion_features.csv")
    df["frame"] = df["frame"].astype(int)
    return df.set_index("frame").sort_index()


def miss_clusters(frames: np.ndarray, all_frames: np.ndarray) -> list[dict]:
    missing = np.setdiff1d(all_frames, frames)
    if len(missing) == 0:
        return []
    splits = np.split(missing, np.where(np.diff(missing) > 1)[0] + 1)
    return [
        {"first": int(s[0]), "last": int(s[-1]), "n": int(len(s))}
        for s in splits
    ]


def adjacent_pairs(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Пары значений ряда на строго соседних кадрах (индекс: шаг 1)."""
    a = s.dropna()
    step = a.index.values[1:] - a.index.values[:-1]
    m = step == 1
    return a.values[:-1][m], a.values[1:][m]


def mean_abs_diff(s: pd.Series) -> float:
    x, y = adjacent_pairs(s)
    return float(np.abs(y - x).mean()) if len(x) else float("nan")


def autocorr_strict(s: pd.Series) -> float:
    x, y = adjacent_pairs(s)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def main() -> None:
    series = {m: load_series(m) for m in MODELS}
    full_range = series[MODELS[0]].index.values

    # --- 1. Сводная таблица по группам ---
    stats_rows = []
    for m in MODELS:
        df = series[m]
        for g in GROUPS:
            s = df[g]
            valid = s.dropna()
            stats_rows.append({
                "model": m,
                "group": g,
                "mean": round(float(valid.mean()), 4),
                "median": round(float(valid.median()), 4),
                "D_j": round(mean_abs_diff(s), 4),
                "valid_pct": round(100.0 * len(valid) / len(df), 1),
                "autocorr_lag1": round(autocorr_strict(s), 3),
            })
    stats = pd.DataFrame(stats_rows)

    # --- 2. Доля детекции и кластеры пропусков ---
    # Детекция = человек найден (строка keypoints.csv не целиком NaN),
    # до порога 0.3; валидность m_j после порога — отдельная величина.
    det_rows = []
    for m in MODELS:
        df = series[m]
        kp = pd.read_csv(RESULTS / m / "keypoints.csv")
        conf_cols = [c for c in kp.columns if c.startswith("c")]
        det = ~kp[conf_cols].isna().all(axis=1)
        det.index = df.index
        det_rows.append({
            "model": m,
            "frames_total": int(len(df)),
            "frames_detected": int(det.sum()),
            "detection_pct": round(100.0 * det.mean(), 1),
            "miss_clusters": miss_clusters(df.index[det].values, full_range),
        })
    detection = pd.DataFrame(det_rows)

    # --- 2б. Статистики по общему подмножеству кадров ---
    # Пересечение кадров, где группа j валидна у ВСЕХ моделей:
    # убирает смещение от селективности детекции/дропаутов точек.
    common_rows = []
    for g in GROUPS:
        mask = np.ones(len(full_range), dtype=bool)
        for m in MODELS:
            mask &= series[m][g].notna().reindex(full_range, fill_value=False).values
        common_idx = full_range[mask]
        for m in MODELS:
            s = series[m][g].reindex(common_idx)
            common_rows.append({
                "group": g,
                "model": m,
                "n_common": len(common_idx),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "D_j": round(mean_abs_diff(s), 4),
                "autocorr_lag1": round(autocorr_strict(s), 3),
            })
    common = pd.DataFrame(common_rows)

    # --- 3. Корреляционные матрицы ---
    head_raw = pd.DataFrame({m: series[m]["m_head"] for m in MODELS})
    head_smooth = head_raw.rolling(5).mean()

    def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
        return df.corr().round(3)

    corr_raw = corr_matrix(head_raw)
    corr_smooth = corr_matrix(head_smooth)

    # тот же набор, но по среднему по всем 5 группам
    mean_raw = pd.DataFrame({
        m: series[m][GROUPS].mean(axis=1) for m in MODELS
    })
    mean_smooth = mean_raw.rolling(5).mean()
    corr_mean_raw = corr_matrix(mean_raw)
    corr_mean_smooth = corr_matrix(mean_smooth)

    # --- вывод ---
    pd.set_option("display.width", 160)
    print("=== Сводная статистика по группам (mean/median/D_j/valid%/autocorr) ===")
    print("    (D_j и autocorr — только пары строго соседних кадров)")
    print(stats.to_string(index=False))
    print("\n=== Детекция ===")
    for _, r in detection.iterrows():
        cl = "; ".join(
            f"{c['first']}-{c['last']} ({c['n']} кадров)" for c in r["miss_clusters"]
        ) or "нет"
        print(f"{r['model']:<11} {r['frames_detected']}/{r['frames_total']} "
              f"({r['detection_pct']}%), пропуски: {cl}")
    print("\n=== По общему подмножеству кадров (пересечение валидных) ===")
    print("    (D_j и autocorr — только пары строго соседних кадров)")
    for g in GROUPS:
        sub = common[common["group"] == g]
        n = sub["n_common"].iloc[0]
        print(f"-- {g} (n={n}) --")
        print(sub.drop(columns=["group", "n_common"]).to_string(index=False))
    print("\n=== Корреляция m_head, сырые ряды ===")
    print(corr_raw.to_string())
    print("\n=== Корреляция m_head, сглаживание окно 5 ===")
    print(corr_smooth.to_string())
    print("\n=== Корреляция mean по 5 группам, сырые ===")
    print(corr_mean_raw.to_string())
    print("\n=== Корреляция mean по 5 группам, сглаживание окно 5 ===")
    print(corr_mean_smooth.to_string())

    out = {
        "stats": stats.to_dict("records"),
        "detection": detection.to_dict("records"),
        "common_subset": common.to_dict("records"),
        "corr_m_head_raw": corr_raw.to_dict(),
        "corr_m_head_smooth_w5": corr_smooth.to_dict(),
        "corr_mean5_raw": corr_mean_raw.to_dict(),
        "corr_mean5_smooth_w5": corr_mean_smooth.to_dict(),
    }
    out_path = RESULTS / "cross_model_summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()

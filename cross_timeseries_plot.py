"""Сводный график главы 4: компонента m_head четырёх моделей (сырые + сглаженные ряды).

Читает results/{model}/motion_features.csv.
Сохраняет results/cross_motion_timeseries.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).parent / "results"
MODELS = ["blazepose", "rtmpose", "rtmo", "openpose"]
COLORS = {
    "blazepose": "tab:blue",
    "rtmpose": "tab:red",
    "rtmo": "tab:green",
    "openpose": "tab:orange",
}
LABELS = {
    "blazepose": "BlazePose",
    "rtmpose": "RTMPose",
    "rtmo": "RTMO",
    "openpose": "OpenPose",
}

PARTERRE = (529 / 25.0, 591 / 25.0)  # с 21.2 по 23.6 с


def main() -> None:
    series = {
        m: pd.read_csv(RESULTS / m / "motion_features.csv").set_index("frame")["m_head"]
        for m in MODELS
    }
    t = pd.read_csv(RESULTS / MODELS[0] / "motion_features.csv")["time_s"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for m in MODELS:
        axes[0].plot(t, series[m], lw=0.8, color=COLORS[m], label=LABELS[m], alpha=0.9)
        axes[1].plot(t, series[m].rolling(5).mean(), lw=1.4, color=COLORS[m], label=LABELS[m])

    axes[0].set_title(
        "Компонента m_head временного представления движения: сопоставление моделей",
        fontsize=11,
    )
    axes[0].set_ylabel("сырые ряды, доли длины торса/кадр", fontsize=9)
    axes[1].set_ylabel("сглаживание окном 5 кадров (0.2 с)", fontsize=9)
    axes[1].set_xlabel("время, с")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=4, loc="upper left")
        ax.axvspan(*PARTERRE, color="gray", alpha=0.15)
    axes[0].annotate("эпизод партера", xy=(PARTERRE[0] + 0.1, axes[0].get_ylim()[1] * 0.75),
                     fontsize=9, color="dimgray")

    fig.tight_layout()
    out = RESULTS / "cross_motion_timeseries.png"
    fig.savefig(out, dpi=150)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()

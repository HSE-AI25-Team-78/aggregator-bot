from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from ..paths import BASELINE_RESULTS_DIR

RESULTS_PATH = BASELINE_RESULTS_DIR / "baseline_results.csv"
PLOT_PATH = BASELINE_RESULTS_DIR / "baseline_f1.png"


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Не найден файл с результатами: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)

    if "model" not in df.columns or "f1_macro" not in df.columns:
        raise ValueError("В baseline_results.csv должны быть колонки 'model' и 'f1_macro'")

    df = df.sort_values(by="f1_macro", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["f1_macro"])
    plt.ylabel("F1-macro")
    plt.title("Сравнение baseline-моделей по F1-macro")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"[+] График baseline-моделей сохранён в: {PLOT_PATH}")


if __name__ == "__main__":
    main()

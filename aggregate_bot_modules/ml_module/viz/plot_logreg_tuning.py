from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from ..paths import EXPERIMENTS_RESULTS_DIR

TUNING_PATH = EXPERIMENTS_RESULTS_DIR / "logreg_tuning.csv"
PLOT_PATH = EXPERIMENTS_RESULTS_DIR / "logreg_tuning_f1.png"


def main():
    if not TUNING_PATH.exists():
        raise FileNotFoundError(f"Не найден файл с тюнингом: {TUNING_PATH}")

    df = pd.read_csv(TUNING_PATH)

    required_cols = {"C", "ngram_range", "f1_macro"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"В logreg_tuning.csv должны быть колонки: {required_cols}")

    df_agg = (
        df.groupby(["C", "ngram_range"], as_index=False)["f1_macro"]
        .max()
        .sort_values(["ngram_range", "C"])
    )

    print("[*] Сводная таблица по C и ngram_range:")
    print(df_agg)

    plt.figure(figsize=(8, 5))

    for ngr in df_agg["ngram_range"].unique():
        sub = df_agg[df_agg["ngram_range"] == ngr].sort_values("C")
        plt.plot(sub["C"], sub["f1_macro"], marker="o", label=f"ngram={ngr}")

    plt.xlabel("C (параметр регуляризации)")
    plt.ylabel("F1-macro")
    plt.title("Тюнинг Logistic Regression: F1 в зависимости от C и ngram_range")
    plt.grid(alpha=0.3)
    plt.legend(title="n-граммы")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"[+] График тюнинга логистической регрессии сохранён в: {PLOT_PATH}")


if __name__ == "__main__":
    main()

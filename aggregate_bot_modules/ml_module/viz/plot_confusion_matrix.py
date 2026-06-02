from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from ..paths import FINAL_RESULTS_DIR, LOGREG_MODELS_DIR, TEST_PATH

TFIDF_PATH = LOGREG_MODELS_DIR / "tfidf.joblib"
MODEL_PATH = LOGREG_MODELS_DIR / "logreg_model.joblib"
LE_PATH = LOGREG_MODELS_DIR / "label_encoder.joblib"
PLOT_PATH = FINAL_RESULTS_DIR / "confusion_matrix.png"


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Не найден test.csv: {TEST_PATH}")

    if not TFIDF_PATH.exists() or not MODEL_PATH.exists() or not LE_PATH.exists():
        raise FileNotFoundError(
            "Не найдены сохранённые модель/TF-IDF/LabelEncoder. Сначала запусти ml.train_final_model"
        )

    df_test = pd.read_csv(TEST_PATH)
    text_col = "text_clean" if "text_clean" in df_test.columns else "text"

    tfidf = joblib.load(TFIDF_PATH)
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)

    X_test = tfidf.transform(df_test[text_col].astype(str))
    y_true = le.transform(df_test["topic"])
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_true, y_pred)
    classes = list(le.classes_)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_ylabel("Истинный класс")
    ax.set_xlabel("Предсказанный класс")
    ax.set_title("Confusion matrix для финальной модели")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"[+] Матрица ошибок сохранена в: {PLOT_PATH}")


if __name__ == "__main__":
    main()

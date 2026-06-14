from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from .paths import FINAL_RESULTS_DIR, LOGREG_MODELS_DIR, ensure_artifact_dirs

TFIDF_PATH = LOGREG_MODELS_DIR / "tfidf.joblib"
MODEL_PATH = LOGREG_MODELS_DIR / "logreg_model.joblib"
LE_PATH = LOGREG_MODELS_DIR / "label_encoder.joblib"
OUT_PATH = FINAL_RESULTS_DIR / "top_words_per_class.csv"


def main(top_n: int = 20) -> None:
    """
    Строим "feature importance" для логистической регрессии:
    топ-N слов для каждой темы (topic), которые сильнее всего тянут
    предсказание в сторону этого класса.
    """

    ensure_artifact_dirs()
    if not TFIDF_PATH.exists() or not MODEL_PATH.exists() or not LE_PATH.exists():
        raise FileNotFoundError(
            "Не найдены tfidf/model/label_encoder. "
            "Сначала запусти ml.train_final_model"
        )

    # Загружаем артефакты
    tfidf = joblib.load(TFIDF_PATH)
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)

    if not hasattr(model, "coef_"):
        raise ValueError(
            "У модели нет coef_. Feature importance доступен "
            "для линейных моделей (LogReg, LinearSVC)."
        )

    feature_names = tfidf.get_feature_names_out()
    classes = list(le.classes_)

    coef = model.coef_  # shape: (num_classes, num_features)

    rows = []

    for class_idx, class_name in enumerate(classes):
        class_coef = coef[class_idx]

        # Топ-N "позитивных" слов для этого класса
        top_pos_idx = np.argsort(class_coef)[-top_n:][::-1]

        for rank, feat_idx in enumerate(top_pos_idx, start=1):
            rows.append(
                {
                    "topic": class_name,
                    "rank": rank,
                    "word": feature_names[feat_idx],
                    "weight": float(class_coef[feat_idx]),
                    "direction": "positive",
                }
            )

        # Если захочешь, можно раскомментировать блок ниже,
        # чтобы добавить и "анти-слова" (наиболее негативные веса).
        #
        # top_neg_idx = np.argsort(class_coef)[:top_n]
        # for rank, feat_idx in enumerate(top_neg_idx, start=1):
        #     rows.append(
        #         {
        #             "topic": class_name,
        #             "rank": rank,
        #             "word": feature_names[feat_idx],
        #             "weight": float(class_coef[feat_idx]),
        #             "direction": "negative",
        #         }
        #     )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"[+] Топ-{top_n} слов на класс сохранены в: {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main(top_n=20)

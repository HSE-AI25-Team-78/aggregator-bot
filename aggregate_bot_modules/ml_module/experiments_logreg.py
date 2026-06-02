from __future__ import annotations

from itertools import product

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from .paths import EXPERIMENTS_RESULTS_DIR, TRAIN_PATH, VAL_PATH, ensure_artifact_dirs

RESULTS_PATH = EXPERIMENTS_RESULTS_DIR / "logreg_tuning.csv"


def load_data():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Сначала запусти ml.dataset_split")

    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)

    text_col = "text_clean" if "text_clean" in df_train.columns else "text"

    if "topic" not in df_train.columns:
        raise ValueError("В train.csv нет 'topic'")

    return df_train, df_val, text_col


def main():
    ensure_artifact_dirs()
    df_train, df_val, text_col = load_data()

    le = LabelEncoder()
    y_train = le.fit_transform(df_train["topic"])
    y_val = le.transform(df_val["topic"])

    configs = []

    C_list = [0.5, 1.0, 2.0, 3.0]
    ngram_list = [(1, 1), (1, 2)]
    min_df_list = [3, 5]

    for C, ngram_range, min_df in product(C_list, ngram_list, min_df_list):
        print(f"\n[*] C={C}, ngram={ngram_range}, min_df={min_df}")

        tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=20000,
        )

        X_train = tfidf.fit_transform(df_train[text_col].astype(str))
        X_val = tfidf.transform(df_val[text_col].astype(str))

        model = LogisticRegression(
            max_iter=2000,
            C=C,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred, average="macro")
        print(f"[+] F1_macro = {f1:.4f}")

        configs.append(
            {
                "C": C,
                "ngram_range": str(ngram_range),
                "min_df": min_df,
                "f1_macro": f1,
            }
        )

    df_res = pd.DataFrame(configs).sort_values("f1_macro", ascending=False)
    df_res.to_csv(RESULTS_PATH, index=False)
    print(f"\n[+] Результаты тюнинга сохранены: {RESULTS_PATH}")
    print(df_res.head())


if __name__ == "__main__":
    main()

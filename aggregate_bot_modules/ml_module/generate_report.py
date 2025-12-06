from __future__ import annotations

from pathlib import Path
from textwrap import indent

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLIT_DIR = DATA_DIR / "splits"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models" / "logreg_tfidf"

REPORT_PATH = RESULTS_DIR / "report.md"


def section(title: str) -> str:
    return f"\n\n## {title}\n\n"


def format_df_as_md(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Конвертация DataFrame в markdown-таблицу (укороченную)."""
    if len(df) > max_rows:
        df_show = df.head(max_rows)
        note = f"\n\n_Показаны первые {max_rows} строк из {len(df)}._\n"
    else:
        df_show = df
        note = ""

    md_table = df_show.to_markdown(index=False)
    return md_table + note


def add_dataset_info(parts: list[str]) -> None:
    processed_path = DATA_DIR / "processed_posts.csv"
    if not processed_path.exists():
        parts.append(section("Обзор датасета"))
        parts.append("Файл `processed_posts.csv` не найден. Сначала запусти `ml.preprocess`.\n")
        return

    parts.append(section("Обзор датасета"))

    df_proc = pd.read_csv(processed_path)
    n_total = len(df_proc)
    topics = df_proc["topic"].value_counts() if "topic" in df_proc.columns else None

    parts.append(f"- Всего сообщений после предобработки: **{n_total}**\n")

    # размеры сплитов
    train_path = SPLIT_DIR / "train.csv"
    val_path = SPLIT_DIR / "val.csv"
    test_path = SPLIT_DIR / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        parts.append(f"- Train: **{len(df_train)}** строк")
        parts.append(f"- Val: **{len(df_val)}** строк")
        parts.append(f"- Test: **{len(df_test)}** строк\n")

        if "topic" in df_train.columns:
            parts.append("Распределение классов в train:\n\n")
            topic_counts = df_train["topic"].value_counts(normalize=False).rename("count")
            topic_share = df_train["topic"].value_counts(normalize=True).rename("share")
            df_topics = pd.concat([topic_counts, topic_share], axis=1)
            df_topics["share"] = (df_topics["share"] * 100).round(2)
            df_topics.reset_index(inplace=True)
            df_topics.rename(columns={"index": "topic", "share": "share_%"},
                             inplace=True)
            parts.append(format_df_as_md(df_topics))
    else:
        parts.append("- Сплиты train/val/test ещё не созданы.\n")

    if topics is not None:
        parts.append("\nРаспределение тем в processed_posts:\n\n")
        df_topics_all = topics.rename("count").reset_index()
        df_topics_all.rename(columns={"index": "topic"}, inplace=True)
        parts.append(format_df_as_md(df_topics_all))


def add_baseline_results(parts: list[str]) -> None:
    path = RESULTS_DIR / "baseline" / "baseline_results.csv"
    parts.append(section("Baseline-модели"))

    if not path.exists():
        parts.append("Файл `baseline_results.csv` не найден. Сначала запусти `ml.baseline_models`.\n")
        return

    df = pd.read_csv(path)
    df_sorted = df.sort_values("f1_macro", ascending=False)

    parts.append("Сводная таблица baseline-моделей:\n\n")
    parts.append(format_df_as_md(df_sorted))

    best = df_sorted.iloc[0]
    parts.append(
        f"\nЛучшая baseline-модель по F1-macro: **{best['model']}** "
        f"с F1-macro = **{best['f1_macro']:.4f}**.\n"
    )


def add_logreg_tuning(parts: list[str]) -> None:
    path = RESULTS_DIR / "experiments" / "logreg_tuning.csv"
    parts.append(section("Тюнинг логистической регрессии"))

    if not path.exists():
        parts.append("Файл `logreg_tuning.csv` не найден. Сначала запусти `ml.experiments_logreg`.\n")
        return

    df = pd.read_csv(path).sort_values("f1_macro", ascending=False)

    parts.append("Топ-10 конфигураций по F1-macro:\n\n")
    parts.append(format_df_as_md(df, max_rows=10))


def add_vectorizers_comparison(parts: list[str]) -> None:
    path = RESULTS_DIR / "experiments" / "vectorizers.csv"
    parts.append(section("Сравнение методов векторизации"))

    if not path.exists():
        parts.append("Файл `vectorizers.csv` не найден. Сначала запусти `ml.experiments_vectorizers`.\n")
        return

    df = pd.read_csv(path).sort_values("f1_macro", ascending=False)
    parts.append("Результаты сравнения различных схем векторизации и моделей:\n\n")
    parts.append(format_df_as_md(df))


def add_model_efficiency(parts: list[str]) -> None:
    path = RESULTS_DIR / "experiments" / "model_efficiency.csv"
    parts.append(section("Производительность моделей (train / inference)"))

    if not path.exists():
        parts.append("Файл `model_efficiency.csv` не найден. Сначала запусти `ml.model_efficiency`.\n")
        return

    df = pd.read_csv(path)
    parts.append("Замер времени обучения и предсказания моделей:\n\n")
    parts.append(format_df_as_md(df))


def add_final_model_metrics(parts: list[str]) -> None:
    parts.append(section("Финальная модель: качество на test"))

    tfidf_path = MODELS_DIR / "tfidf.joblib"
    model_path = MODELS_DIR / "logreg_model.joblib"
    le_path = MODELS_DIR / "label_encoder.joblib"
    test_path = SPLIT_DIR / "test.csv"

    if not (tfidf_path.exists() and model_path.exists() and le_path.exists() and test_path.exists()):
        parts.append(
            "Не найдены артефакты финальной модели или test.csv. "
            "Сначала запусти `ml.train_final_model` и `ml.dataset_split`.\n"
        )
        return

    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    df_test = pd.read_csv(test_path)

    text_col = "text_clean" if "text_clean" in df_test.columns else "text"
    X_test = tfidf.transform(df_test[text_col].astype(str))
    y_true = le.transform(df_test["topic"])

    y_pred = model.predict(X_test)

    f1 = f1_score(y_true, y_pred, average="macro")
    parts.append(f"- F1-macro на test: **{f1:.4f}**\n")

    report = classification_report(y_true, y_pred, target_names=list(le.classes_))
    parts.append("\nКлассификационный отчёт по классам:\n\n")
    parts.append("```text\n" + report + "\n```")


def add_top_words(parts: list[str]) -> None:
    path = RESULTS_DIR / "final" / "top_words_per_class.csv"
    parts.append(section("Наиболее важные слова для каждой темы (feature importance)"))

    if not path.exists():
        parts.append("Файл `top_words_per_class.csv` не найден. Сначала запусти `ml.feature_importance`.\n")
        return

    df = pd.read_csv(path)

    for topic in df["topic"].unique():
        sub = df[(df["topic"] == topic) & (df["direction"] == "positive")].sort_values("rank")
        parts.append(f"\n**Тема: {topic}**\n\n")
        # показываем 10 топ-слов, если их больше
        sub_show = sub.head(10)[["rank", "word", "weight"]]
        parts.append(format_df_as_md(sub_show))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append("# Отчёт по ML-части проекта бота-агрегатора\n")

    add_dataset_info(parts)
    add_baseline_results(parts)
    add_logreg_tuning(parts)
    add_vectorizers_comparison(parts)
    add_model_efficiency(parts)
    add_final_model_metrics(parts)
    add_top_words(parts)

    report_text = "\n".join(parts)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"[+] Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
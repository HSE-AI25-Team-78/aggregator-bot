from __future__ import annotations

from . import preprocess
from . import dataset_split
from . import baseline_models
from . import experiments_logreg
from . import train_final_model
from . import experiments_vectorizers
from . import model_efficiency
from . import feature_importance
from .viz import plot_baseline_results, plot_confusion_matrix, plot_logreg_tuning
from .paths import PROCESSED_PATH, RAW_PATH, SPLIT_DIR, TEST_PATH, TRAIN_PATH, VAL_PATH, ensure_artifact_dirs


def step_preprocess():
    if PROCESSED_PATH.exists():
        print(f"[skip] {PROCESSED_PATH} уже существует, пропускаем preprocess")
        return
    print("\n=== Шаг 1: предобработка текстов (preprocess) ===")
    preprocess.main(min_len=15)


def step_split():
    if TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists():
        print(f"[skip] Файлы в {SPLIT_DIR} уже есть, пропускаем split")
        return
    print("\n=== Шаг 2: train/val/test split ===")
    dataset_split.main()


def step_baseline():
    print("\n=== Шаг 3: baseline-модели ===")
    baseline_models.main()


def step_logreg_tuning():
    print("\n=== Шаг 4: тюнинг логистической регрессии ===")
    experiments_logreg.main()


def step_train_final():
    print("\n=== Шаг 5: обучение финальной модели ===")
    train_final_model.main()


def step_visualizations():
    print("\n=== Шаг 6: базовые визуализации ===")
    try:
        plot_baseline_results.main()
    except Exception as e:
        print(f"[warn] Не удалось построить baseline plot: {e}")

    try:
        plot_logreg_tuning.main()
    except Exception as e:
        print(f"[warn] Не удалось построить график тюнинга: {e}")

    try:
        plot_confusion_matrix.main()
    except Exception as e:
        print(f"[warn] Не удалось построить confusion matrix: {e}")


def step_advanced_experiments():
    print("\n=== Шаг 7: расширенные эксперименты (vectorizers, efficiency, feature importance) ===")

    # 7.1. Сравнение разных векторизаций
    try:
        print("\n--- 7.1 experiments_vectorizers ---")
        experiments_vectorizers.main()
    except Exception as e:
        print(f"[warn] Ошибка в experiments_vectorizers: {e}")

    # 7.2. Измерение эффективности моделей
    try:
        print("\n--- 7.2 model_efficiency ---")
        model_efficiency.main()
    except Exception as e:
        print(f"[warn] Ошибка в model_efficiency: {e}")

    # 7.3. Топ-слова для каждой темы (feature importance)
    try:
        print("\n--- 7.3 feature_importance ---")
        feature_importance.main(top_n=20)
    except Exception as e:
        print(f"[warn] Ошибка в feature_importance: {e}")


def main():
    ensure_artifact_dirs()
    print("=== Запуск полного ML-пайплайна для бота-агрегатора ===")
    print(f"Канонический датасет: {RAW_PATH}")

    step_preprocess()
    step_split()
    step_baseline()
    step_logreg_tuning()
    step_train_final()
    step_visualizations()
    step_advanced_experiments()

    print("\n=== Готово. Все шаги пайплайна выполнены. ===")


if __name__ == "__main__":
    main()

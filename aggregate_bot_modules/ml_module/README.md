

# ML Module - Machine Learning Pipeline

Этот модуль отвечает за **полный цикл обработки данных и обучения моделей**.
Внутри реализован воспроизводимый ML-пайплайн, включающий предобработку текстов, разбиение выборок, обучение baseline-моделей, тюнинг, обучение финальной модели и построение визуализаций.

---

##  Структура

```
aggregate_bot_modules/ml_module/
│
├── preprocess.py              # Шаг 1: очистка текстов и подготовка processed_posts.csv
├── dataset_split.py           # Шаг 2: stratified train/val/test split
├── baseline_models.py         # Шаг 3: обучение baseline-моделей (LogReg, SVM, NB, KNN)
├── experiments_logreg.py      # Шаг 4: тюнинг Logistic Regression по C, ngram_range, min_df
├── train_final_model.py       # Шаг 5: обучение финальной модели и сохранение артефактов
│
├── run_all.py                 # ЕДИНАЯ ТОЧКА ВХОДА
│
└── viz/                       # Визуализации результатов
    ├── plot_baseline_results.py
    ├── plot_logreg_tuning.py
    ├── plot_confusion_matrix.py
    └── __init__.py
```

Канонический размеченный датасет:

```text
ML/raw_posts_labeled.csv
```

Все воспроизводимые артефакты пайплайна теперь складываются в:

```text
ML/artifacts/
```

Боевые артефакты для сервиса экспортируются в:

```text
service/config/
```

Локальный MLflow tracking для финального обучения складывается в:

```text
ML/mlruns/
```

---

##  Основной сценарий использования

Обычно для запуска всего ML-пайплайна достаточно **одной команды**:

```bash
python -m ml.run_all
```

Скрипт автоматически выполняет:

1. **Предобработку**
2. **Разбиение данных**
3. **Baseline-модели**
4. **Тюнинг Logistic Regression**
5. **Финальную модель**
6. **Визуализации**

7. **Расширенные эксперименты**
   - сравнение различных методов векторизации (TF-IDF, Bag-of-Words, HashingVectorizer);
   - анализ производительности моделей (время обучения и инференса);
   - анализ значимости признаков (feature importance) для логистической регрессии.

---

##  Где искать результаты

```
ML/
  artifacts/
    processed_posts.csv
    splits/
      train.csv
      val.csv
      test.csv
    results/
    baseline/
      baseline_results.csv
      baseline_f1.png

    experiments/
      logreg_tuning.csv
      logreg_tuning_f1.png

    final/
      confusion_matrix.png
```

---

##  Краткое описание ключевых скриптов

### preprocess.py  
Очищает текст, сохраняет processed_posts.csv

### dataset_split.py  
Stratified split: train/val/test

### baseline_models.py  
Обучает базовые ML-модели, сохраняет baseline_results.csv

### experiments_logreg.py  
Перебор C, ngram_range, min_df

### train_final_model.py  
Финальная модель + сохранение TF-IDF/модели/LabelEncoder

Дополнительно этот шаг теперь умеет:
- сохранять итоговые classification reports и confusion matrices;
- писать training summary в `ML/artifacts/results/final/training_summary.json`;
- опционально логировать параметры, метрики и артефакты в локальный MLflow.

### viz/*  
Графики: baseline -> tuning -> confusion matrix

---

##  Требования

```text
scikit-learn
pandas
numpy
matplotlib
joblib
```

---

##  Запуск отдельных шагов

```bash
python -m ml.preprocess
python -m ml.dataset_split
python -m ml.baseline_models
python -m ml.experiments_logreg
python -m ml.train_final_model
python -m ml.viz.plot_baseline_results
python -m ml.viz.plot_logreg_tuning
python -m ml.viz.plot_confusion_matrix
```

### Дополнительные эксперименты

```bash
python -m ml.experiments_vectorizers      # сравнение векторизаций
python -m ml.model_efficiency             # скорость обучения и инференса моделей
python -m ml.feature_importance           # топ-слова по весам логистической регрессии
```

### Локальный MLflow tracking

По умолчанию локальный tracking включён для:

- `aggregate_bot_modules.ml_module.baseline_models`
- `aggregate_bot_modules.ml_module.train_final_model`
- `ML/pseudo_label_pipeline.py`
- `ML/targeted_weak_class_pipeline.py`

Все эти шаги пишут run в один и тот же локальный `ML/mlruns/`, если пакет `mlflow-skinny`/`mlflow` установлен.

Полезные переменные окружения:

```bash
ENABLE_MLFLOW=1
MLFLOW_TRACKING_URI=file:///abs/path/to/ML/mlruns
MLFLOW_EXPERIMENT_NAME=aggregator_bot_training
MLFLOW_RUN_NAME=train_final_model
```

Если нужно полностью выключить tracking:

```bash
ENABLE_MLFLOW=0 python -m aggregate_bot_modules.ml_module.train_final_model
```

Если `mlflow` установлен, UI можно поднять локально так:

```bash
mlflow ui --backend-store-uri file:///ABS_PATH_TO/ML/mlruns
```

После этого открой:

- `http://127.0.0.1:5000`

В интерфейсе появятся отдельные experiments:

- `aggregator_bot_baselines`
- `aggregator_bot_training`
- `aggregator_bot_pseudo_labeling`
- `aggregator_bot_targeted_weak_classes`

---


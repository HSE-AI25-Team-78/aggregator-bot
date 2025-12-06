

# ML Module — Machine Learning Pipeline

Этот модуль отвечает за **полный цикл обработки данных и обучения моделей**.
Внутри реализован воспроизводимый ML-пайплайн, включающий предобработку текстов, разбиение выборок, обучение baseline-моделей, тюнинг, обучение финальной модели и построение визуализаций.

---

##  Структура

```
ml/
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
data/
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

### viz/*  
Графики: baseline → tuning → confusion matrix

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
---


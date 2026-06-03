# 📊 Text Classification Pipeline with MLflow & Hydra

Production-ready text classification pipeline using **TF-IDF + Logistic Regression**, fully tracked with **MLflow**, configured via **Hydra**, and equipped with baseline comparison, error analysis, robustness testing, and PRD model versioning.

---

## 📦 Prerequisites & Setup

### 1. Install Dependencies
```bash
# Using uv (recommended)
uv add hydra-core mlflow boto3 scikit-learn pandas matplotlib joblib omegaconf

# Or pip
pip install hydra-core mlflow boto3 scikit-learn pandas matplotlib joblib omegaconf
```

### 2. Prepare Data
Разместите стратифицированные сплиты в `./splits/`:
```
./splits/
├── train.csv
├── val.csv
└── test.csv
```
Колонки: `text_clean` (или `text`) и `topic`.

---

## 🗂️ Project Structure
```
.
├── config.yaml              # Hydra-конфиг (данные, модель, MLflow, seed)
├── train_with_mlflow.py     # Обучение, метрики, анализ ошибок, robustness, логирование
├── predict_prd.py           # Инференс модели с тегом PRD
└── README.md
```
---

## ⚙️ Configuration (`config.yaml`)
Все параметры вынесены в конфиг и могут переопределяться через CLI без изменения кода:
```yaml
data:
  train_path: ../data/splits/train.csv
  val_path:   ../data/splits/val.csv
  test_path:  ../data/splits/test.csv
  text_col: text_clean
  target_col: topic

tfidf:
  ngram_range: [1, 2]
  min_df: 3
  max_features: 20000

model:
  name: LogisticRegression
  max_iter: 2000
  C: 1.0
  solver: lbfgs
  n_jobs: -1
  random_state: 42

baseline:
  name: MultinomialNB
  alpha: 1.0

mlflow:
  experiment_name: text_classification_prd
  tracking_uri: http://localhost:5050
  s3_endpoint_url: http://localhost:9005
  aws_access_key_id: admin
  aws_secret_access_key: password
  artifact_location: mlflow-artifacts:/

seed: 42
```

---

## 🚀 Usage

### 🔹 Обучение и эксперименты
```bash
# 1. Базовый запуск (параметры из config.yaml)
python train_with_mlflow.py

# 2. Эксперимент с регуляризацией C=0.1
python train_with_mlflow.py model.C=0.1

# 3. Эксперимент с C=5.0 и solver=saga
python train_with_mlflow.py model.C=5.0 model.solver=saga

# 4. Эксперимент с линейкой параметров
python3 train_with_mlflow.py -m model.C=0.01,0.1,0.5,1.0,5.0,10.0
```
📁 Hydra автоматически создаст директорию `outputs/` с логами, конфигами и метаданными каждого запуска.

### 🔹 Инференс PRD-модели
```bash
python predict_prd.py
```
Скрипт автоматически:
1. Находит последний run с тегом `tags.stage = 'PRD'`
2. Загружает `LogisticRegression`, `TfidfVectorizer` и `LabelEncoder` из MinIO
3. Выполняет тестовый предикт на `test.csv`

---

## 📊 MLflow & Experiment Tracking

- **UI:** http://localhost:5050
- **Хранилище артефактов:** MinIO (S3-compatible) → `s3://mlflow-bucket/mlflow/`
- **Логгируется:**
  - Гиперпараметры TF-IDF и модели
  - Метрики `train/val/test` (F1_macro, Accuracy, Precision, Recall)
  - Confusion Matrix (`confusion_matrix.png`)
  - Примеры ошибок (`error_analysis_sample.csv`)
  - Устойчивость к шуму (`robustness_stability_score`)
  - Сериализованные артефакты (`tfidf.joblib`, `label_encoder.joblib`, `model/`)
- **Тегирование:** Финальный run помечается `stage: PRD` для быстрого деплоя.

---

## 🔄 Reproducibility & Versioning

- ✅ `seed=42` зафиксирован для `random`, `numpy`, `sklearn`
- ✅ Все гиперпараметры версионируются через Hydra + MLflow
- ✅ Данные загружаются из фиксированных сплитов `data/splits/`
- ✅ Артефакты привязаны к `run_id` и хранятся в S3/MinIO
- ✅ PRD-модель загружается по тегу, без ручного указания ID

---

## 🛠️ Troubleshooting

| Проблема | Решение |
|----------|---------|
| `ConnectionClosedError / BadStatusLine localhost:9000` | Убедитесь, что в скрипте задано `AWS_S3_ADDRESSING_STYLE=path` и порт в `config.yaml` совпадает с `S3_API_PORT` в `.env` |
| `ModuleNotFoundError: No module named 'hydra'` | Установите `uv add hydra-core` (пакет в PyPI называется `hydra-core`) |
| `ModuleNotFoundError: No module named 'mlflow'` | `uv add mlflow boto3` |
| `.venv` пробрасывается в Docker и ломает окружение | Добавьте в `docker-compose.yml`: `- /app/.venv` в `volumes` |
| `ImportError: attempted relative import` | Запускайте из корня: `python -m aggregate_bot_modules.ml_module.run_all` или используйте абсолютные импорты |

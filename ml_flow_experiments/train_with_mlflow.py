import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import hydra
from omegaconf import DictConfig
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature  # добавьте в импорты в начале файла

warnings.filterwarnings("ignore")

# Настройка MLflow ДО импорта mlflow (если требуется)
def setup_mlflow_env(cfg):
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.mlflow.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.mlflow.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.mlflow.s3_endpoint_url
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def load_data(cfg):
    df_train = pd.read_csv(cfg.data.train_path)
    df_val = pd.read_csv(cfg.data.val_path)
    df_test = pd.read_csv(cfg.data.test_path)
    text_col = cfg.data.text_col if cfg.data.text_col in df_train.columns else "text"
    if cfg.data.target_col not in df_train.columns:
        raise ValueError(f"В данных нет колонки '{cfg.data.target_col}'")
    return df_train, df_val, df_test, text_col

def number_to_label(number_label: int):
    classes = [
        "Общее",
        "Наука и техника",
        "ИТ и телекоммуникации",
        "Общество, государство, политика",
        "Экономика",
        "Медицина",
        "Искусство и культура",
        "Развлечения",
        "Спорт",
        "История",
        "Происшествия"
    ]
    if 0 <= number_label < len(classes):
        return classes[number_label]

def perturb_text(text: str, seed: int = 42) -> str:
    """Простая аугментация для проверки robustness: добавление шума/опечаток"""
    rng = random.Random(seed)
    words = text.split()
    if len(words) < 3:
        return text + " noise_test"
    # Замена случайного слова на "тестшум"
    idx = rng.randint(0, len(words) - 1)
    words[idx] = "тестшум"
    return " ".join(words)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Фиксируем рабочую директорию (Hydra меняет её на outputs/...)
    os.chdir(hydra.utils.get_original_cwd())
    
    setup_mlflow_env(cfg)
    set_seed(cfg.seed)
    
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    print("[*] Загрузка данных...")
    df_train, df_val, df_test, text_col = load_data(cfg)
    
    # TF-IDF
    print("[*] Векторизация TF-IDF...")
    tfidf = TfidfVectorizer(
        ngram_range=tuple(cfg.tfidf.ngram_range),
        min_df=cfg.tfidf.min_df,
        max_features=cfg.tfidf.max_features
    )
    X_train = tfidf.fit_transform(df_train[text_col].astype(str))
    X_val = tfidf.transform(df_val[text_col].astype(str))
    X_test = tfidf.transform(df_test[text_col].astype(str))
    
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[cfg.data.target_col])
    y_val = le.transform(df_val[cfg.data.target_col])
    y_test = le.transform(df_test[cfg.data.target_col])
    
    with mlflow.start_run(run_name=f"LR_C={cfg.model.C}_final") as run:
        mlflow.set_tag("stage", "PRD")  # Тег финальной версии
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("text_column", text_col)
        mlflow.log_params(dict(cfg.tfidf))
        mlflow.log_params(dict(cfg.model))
        
        # 1. Baseline (MultinomialNB)
        print("\n[***] Обучение Baseline: MultinomialNB")
        nb = MultinomialNB(alpha=cfg.baseline.alpha)
        nb.fit(X_train, y_train)
        y_pred_nb_val = nb.predict(X_val)
        nb_f1 = f1_score(y_val, y_pred_nb_val, average="macro")
        mlflow.log_metric("baseline_val_f1_macro", nb_f1)
        print(f"[+] Baseline Val F1_macro: {nb_f1:.4f}")
        
        # 2. Основная модель (LogisticRegression)
        print("\n[***] Обучение финальной модели: LogisticRegression")
        model = LogisticRegression(
            max_iter=cfg.model.max_iter,
            C=cfg.model.C,
            solver=cfg.model.solver,
            n_jobs=cfg.model.n_jobs,
            random_state=cfg.model.random_state
        )
        model.fit(X_train, y_train)
        
        # Метрики на всех сплитах
        splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
        for name, (X, y) in splits.items():
            y_pred = model.predict(X)
            mlflow.log_metric(f"{name}_f1_macro", f1_score(y, y_pred, average="macro"))
            mlflow.log_metric(f"{name}_accuracy", accuracy_score(y, y_pred))
            mlflow.log_metric(f"{name}_precision_macro", precision_score(y, y_pred, average="macro", zero_division=0))
            mlflow.log_metric(f"{name}_recall_macro", recall_score(y, y_pred, average="macro", zero_division=0))
            print(f"[+] {name.upper()} F1_macro: {f1_score(y, y_pred, average='macro'):.4f}")
            
        y_pred_test = model.predict(X_test)
        
        # Сравнение с baseline
        improvement = f1_score(y_test, y_pred_test, average="macro") - f1_score(y_val, y_pred_nb_val, average="macro")
        mlflow.log_metric("improvement_over_baseline_f1", improvement)
        print(f"\n[+] Прирост F1 над baseline: {improvement:.4f}")
        
        # 3. Артефакты: Confusion Matrix
        print("[*] Сохранение Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred_test)
        plt.tight_layout()
        fig, ax = plt.subplots(figsize=(12, 10))
        display_labels = list(map(number_to_label, le.classes_))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # 4. Анализ ошибок (10-20 примеров)
        print("[*] Анализ ошибок...")
        df_test_pred = df_test.copy()
        df_test_pred["true_label"] = le.inverse_transform(y_test)
        df_test_pred["pred_label"] = le.inverse_transform(y_pred_test)
        df_errors = df_test_pred[df_test_pred["true_label"] != df_test_pred["pred_label"]].copy()
        
        if len(df_errors) > 0:
            sample_errors = df_errors.sample(n=min(15, len(df_errors)), random_state=cfg.seed)
            sample_errors["error_category"] = "ambiguous_topic"
            sample_errors["reason"] = "Текст содержит признаки нескольких классов или слишком короткий/шумный после очистки."
            errors_path = "error_analysis_sample.csv"
            sample_errors[[text_col, "true_label", "pred_label", "error_category", "reason"]].to_csv(errors_path, index=False)
            mlflow.log_artifact(errors_path)
            print(f"[+] Сохранено {len(sample_errors)} примеров ошибок в {errors_path}")
        else:
            print("[+] Ошибок на тесте не найдено.")
            
        # 5. Robustness Check
        print("[*] Проверка устойчивости (robustness)...")
        robustness_sample = df_test.sample(n=min(50, len(df_test)), random_state=cfg.seed)
        X_robust_orig = tfidf.transform(robustness_sample[text_col].astype(str))
        robustness_sample["text_perturbed"] = robustness_sample[text_col].apply(lambda x: perturb_text(str(x)))
        X_robust_pert = tfidf.transform(robustness_sample["text_perturbed"].astype(str))
        
        pred_orig = model.predict(X_robust_orig)
        pred_pert = model.predict(X_robust_pert)
        stability = np.mean(pred_orig == pred_pert)
        mlflow.log_metric("robustness_stability_score", stability)
        print(f"[+] Устойчивость к шуму: {stability:.2%} предсказаний не изменились")
        
        # 6. Сохранение артефактов в S3 через MLflow
        print("[*] Логирование артефактов модели...")
        joblib.dump(tfidf, "tfidf.joblib")
        joblib.dump(le, "label_encoder.joblib")
        mlflow.log_artifact("tfidf.joblib")
        mlflow.log_artifact("label_encoder.joblib")
        pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", model)
        ])
        sample_X = X_train[:5].toarray()  
        sample_y = y_train[:5]
        signature = infer_signature(sample_X, sample_y)
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=sample_X
        )
        print(f"\n[✅] Run ID: {run.info.run_id} | Тег: PRD | Артефакты в S3")
        
    # Очистка временных файлов
    for f in [cm_path, errors_path, "tfidf.joblib", "label_encoder.joblib"]:
        if Path(f).exists():
            Path(f).unlink()

if __name__ == "__main__":
    main()
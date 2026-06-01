import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import hydra
from omegaconf import DictConfig

def setup_mlflow_env(cfg):
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.mlflow.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.mlflow.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.mlflow.s3_endpoint_url
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

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

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    setup_mlflow_env(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # Поиск run с тегом PRD
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'PRD'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise RuntimeError("Не найдено ни одного run с тегом 'PRD'")
        
    prd_run = runs[0]
    run_id = prd_run.info.run_id
    print(f"[+] Загрузка PRD модели из run: {run_id}")
    
    # Загрузка артефактов
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    tfidf_path = client.download_artifacts(run_id, "tfidf.joblib")
    le_path = client.download_artifacts(run_id, "label_encoder.joblib")
    
    tfidf = joblib.load(tfidf_path)
    le = joblib.load(le_path)
    
    # Тестовый предикт
    df_test = pd.read_csv(cfg.data.test_path)
    text_col = cfg.data.text_col if cfg.data.text_col in df_test.columns else "text"
    sample = df_test.head(5)
    
    X_sample = tfidf.transform(sample[text_col].astype(str))
    preds = model.predict(X_sample)
    pred_labels = le.inverse_transform(preds)
    
    print("\n[🔮] Тестовые предсказания PRD-модели:")
    for i, row in sample.iterrows():
        print(f"Text: {str(row[text_col])[:60]}... | Pred: {number_to_label(pred_labels[i])}")
        
if __name__ == "__main__":
    main()
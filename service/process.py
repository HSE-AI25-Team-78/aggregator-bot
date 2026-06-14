import re
from scipy.sparse._matrix import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from pathlib import Path
import json
import warnings
from sklearn.exceptions import InconsistentVersionWarning

SERVICE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SERVICE_DIR / 'config'
MANIFEST_PATH = CONFIG_PATH / 'model_manifest.json'
UNCERTAIN_LABEL = 'Неуверенно'
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.24
TOPICAL_THRESHOLD = 0.55
GENERAL_THRESHOLD = 0.35

def pickle_read(object, file_path):
    with open(file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

def get_model():
    manifest = load_model_manifest() or {}
    model_path = manifest.get('artifacts', {}).get('model')
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return pickle_read(None, candidate)
    model_file_path = CONFIG_PATH / 'models' / 'MultinomialNB.pkl'
    return pickle_read(MultinomialNB(), model_file_path)

def get_label_encoder() -> LabelEncoder:
    manifest = load_model_manifest() or {}
    path = manifest.get('artifacts', {}).get('label_encoder')
    if path:
        candidate = Path(path)
        if candidate.exists():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return pickle_read(None, candidate)
    return pickle_read(LabelEncoder(), CONFIG_PATH / 'label_encoder.pkl')

def get_vectorizer() -> TfidfVectorizer:
    manifest = load_model_manifest() or {}
    path = manifest.get('artifacts', {}).get('vectorizer')
    if path:
        candidate = Path(path)
        if candidate.exists():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return pickle_read(None, candidate)
    return pickle_read(TfidfVectorizer(), CONFIG_PATH / 'vectorizer.pkl')


def get_topicality_gate():
    gate_model_path = CONFIG_PATH / 'topicality_gate.pkl'
    gate_vectorizer_path = CONFIG_PATH / 'topicality_gate_vectorizer.pkl'
    if not gate_model_path.exists() or not gate_vectorizer_path.exists():
        return None
    gate_model = pickle_read(LogisticRegression(), gate_model_path)
    gate_vectorizer = pickle_read(TfidfVectorizer(), gate_vectorizer_path)
    return gate_vectorizer, gate_model


def get_model_artifact_status() -> dict:
    manifest = load_model_manifest() or {}
    model_path = Path(manifest.get('artifacts', {}).get('model', CONFIG_PATH / 'models' / 'MultinomialNB.pkl'))
    vectorizer_path = Path(manifest.get('artifacts', {}).get('vectorizer', CONFIG_PATH / 'vectorizer.pkl'))
    label_encoder_path = Path(manifest.get('artifacts', {}).get('label_encoder', CONFIG_PATH / 'label_encoder.pkl'))
    gate_model_path = CONFIG_PATH / 'topicality_gate.pkl'
    gate_vectorizer_path = CONFIG_PATH / 'topicality_gate_vectorizer.pkl'
    return {
        'model': model_path.exists(),
        'vectorizer': vectorizer_path.exists(),
        'label_encoder': label_encoder_path.exists(),
        'topicality_gate_model': gate_model_path.exists(),
        'topicality_gate_vectorizer': gate_vectorizer_path.exists(),
        'manifest': MANIFEST_PATH.exists(),
        'paths': {
            'model': str(model_path.resolve()),
            'vectorizer': str(vectorizer_path.resolve()),
            'label_encoder': str(label_encoder_path.resolve()),
            'topicality_gate_model': str(gate_model_path.resolve()),
            'topicality_gate_vectorizer': str(gate_vectorizer_path.resolve()),
            'manifest': str(MANIFEST_PATH.resolve()),
        },
    }


def load_model_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))


def clean_text(text: str) -> str:
    """
    Простая очистка текста:
      - убираем ссылки, @упоминания, #хэштеги
      - приводим к нижнему регистру
      - выкидываем цифры, лишние символы и пунктуацию
      - схлопываем лишние пробелы
    """
    URL_RE = re.compile(r"http\S+|www\.\S+|t\.me/\S+")
    MENTION_RE = re.compile(r"@\w+")
    HASHTAG_RE = re.compile(r"#\w+")
    NON_LETTERS_RE = re.compile(r"[^a-zA-Zа-яА-ЯёЁ\s]")

    if not isinstance(text, str):
        return ""

    # нижний регистр
    t = text.lower()

    # ссылки, упоминания, хэштеги
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)

    # убираем всё, кроме букв и пробелов
    t = NON_LETTERS_RE.sub(" ", t)

    # схлопываем пробелы
    t = re.sub(r"\s+", " ", t).strip()

    return t


def vectorize_text(text: str) -> spmatrix:
    tfidf = get_vectorizer()
    return tfidf.transform([text])


def run_pipeline(text: str) -> str | None:
    result = run_pipeline_with_meta(text)
    return result['label']


def run_pipeline_with_meta(text: str) -> dict:
    cleaned_text = clean_text(text)
    gate = get_topicality_gate()
    gate_probability = None
    if gate:
        gate_vectorizer, gate_model = gate
        gate_features = gate_vectorizer.transform([cleaned_text])
        gate_probability = float(gate_model.predict_proba(gate_features)[0][1])
        if gate_probability <= GENERAL_THRESHOLD:
            return {
                'label': 'Общее',
                'confidence': gate_probability,
                'clean_text': cleaned_text,
                'topical_probability': gate_probability,
            }
        if gate_probability < TOPICAL_THRESHOLD:
            return {
                'label': UNCERTAIN_LABEL,
                'confidence': gate_probability,
                'clean_text': cleaned_text,
                'topical_probability': gate_probability,
            }
    vectorized_text = vectorize_text(cleaned_text)
    
    model = get_model()
    raw_label = model.predict(vectorized_text)
    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = float(np.max(probabilities))
    
    le = get_label_encoder()
    label = str(le.inverse_transform(raw_label)[0])

    if label.isdigit():
        label = number_to_label(int(label))
    if confidence is not None and confidence < UNCERTAIN_CONFIDENCE_THRESHOLD:
        label = UNCERTAIN_LABEL
    return {
        'label': label,
        'confidence': confidence,
        'clean_text': cleaned_text,
        'topical_probability': gate_probability,
    }


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

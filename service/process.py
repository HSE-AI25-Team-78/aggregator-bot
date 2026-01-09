import re
from scipy.sparse._matrix import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from pathlib import Path

CONFIG_PATH = Path('./config')

def pickle_read(object, file_path):
    with open(file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

def get_model() -> MultinomialNB:
    MODEL_FILE_PATH = CONFIG_PATH / 'models' / 'MultinomialNB.pkl'
    return pickle_read(MultinomialNB(), MODEL_FILE_PATH)

def get_label_encoder() -> LabelEncoder:
    return pickle_read(LabelEncoder(), CONFIG_PATH / 'label_encoder.pkl')

def get_vectorizer() -> TfidfVectorizer:
    return pickle_read(TfidfVectorizer(), CONFIG_PATH / 'vectorizer.pkl')


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
    cleaned_text = clean_text(text)
    vectorized_text = vectorize_text(cleaned_text)
    
    model = get_model()
    label = model.predict(vectorized_text)
    
    le = get_label_encoder()
    label = str(le.inverse_transform(label)[0])

    if label.isdigit():
        label = number_to_label(int(label))
    return label


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

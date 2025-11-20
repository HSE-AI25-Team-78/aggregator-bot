import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc,
    NamesExtractor
)

def load_russian():
    # Для русского языка
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except:
        pass

def clean_telegram_text(text, remove_links=True, remove_formatting=True, remove_emoji=True, remove_mentions=True, remove_hashtags=False):
    """
    Очищает текст Telegram сообщения от Markdown разметки и других элементов
    
    Параметры:
    text (str): исходный текст
    remove_links (bool): удалять ли ссылки
    remove_formatting (bool): удалять ли Markdown форматирование (жирный, курсив и т.д.)
    remove_emoji (bool): удалять ли эмодзи
    remove_mentions (bool): удалять ли упоминания (@username)
    remove_hashtags (bool): удалять ли хэштеги
    
    Возвращает:
    str: очищенный текст
    """
    if not isinstance(text, str):
        return text
    
    cleaned_text = text
    
    # Удаление ссылок [текст](URL) -> текст
    if remove_links:
        cleaned_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned_text)
    
    # Удаление Markdown форматирования
    if remove_formatting:
        # Жирный текст **текст** -> текст
        cleaned_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_text)
        # Курсив *текст* -> текст
        cleaned_text = re.sub(r'\*([^*]+)\*', r'\1', cleaned_text)
        # Курсив _текст_ -> текст
        cleaned_text = re.sub(r'_([^_]+)_', r'\1', cleaned_text)
        # Подчеркивание __текст__ -> текст
        cleaned_text = re.sub(r'__([^_]+)__', r'\1', cleaned_text)
        # Зачеркивание ~~текст~~ -> текст
        cleaned_text = re.sub(r'~~([^~]+)~~', r'\1', cleaned_text)
        # Код `код` -> код
        cleaned_text = re.sub(r'`([^`]+)`', r'\1', cleaned_text)
    
    # Удаление эмодзи и специальных символов
    if remove_emoji:
        # Паттерн для эмодзи и специальных символов
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # эмоции
            u"\U0001F300-\U0001F5FF"  # символы и пиктограммы
            u"\U0001F680-\U0001F6FF"  # транспорт и карты
            u"\U0001F1E0-\U0001F1FF"  # флаги
            u"\U00002500-\U00002BEF"  # различные символы
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # вариационные селекторы
            u"\u3030"
            "]+", flags=re.UNICODE)
        cleaned_text = emoji_pattern.sub(r'', cleaned_text)
    
    # Удаление упоминаний
    if remove_mentions:
        cleaned_text = re.sub(r'@\w+', '', cleaned_text)
    
    # Удаление хэштегов
    if remove_hashtags:
        cleaned_text = re.sub(r'#\w+', '', cleaned_text)
    
    # Удаление лишних пробелов и переносов строк
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# Применение функции к DataFrame
def apply_text_cleaning(df, text_column='text', **kwargs):
    """
    Применяет очистку текста к указанному столбцу DataFrame
    
    Параметры:
    df (pd.DataFrame): исходный DataFrame
    text_column (str): название столбца с текстом
    **kwargs: параметры для функции clean_telegram_text
    
    Возвращает:
    pd.DataFrame: DataFrame с очищенным текстом
    """
    df_cleaned = df.copy()
    df_cleaned[f'{text_column}_cleaned'] = df_cleaned[text_column].apply(
        lambda x: clean_telegram_text(x, **kwargs)
    )
    return df_cleaned



# # Инициализация компонентов Natasha
# segmenter = Segmenter()
# morph_vocab = MorphVocab()
# emb = NewsEmbedding()
# morph_tagger = NewsMorphTagger(emb)
# syntax_parser = NewsSyntaxParser(emb)
# ner_tagger = NewsNERTagger(emb)
# names_extractor = NamesExtractor(morph_vocab)



def get_stopwords():
    russian_stopwords = set(stopwords.words('russian'))
    custom_stopwords = set([
        'рбк', 'telegram', 'reuters', 'com', 'ru', 'kod', 'канал', 'канале', 'тасс', 'max',
        'подписывайтесь', 'подписаться', 'подпишись', 'подпишитесь', 'бусты', 'подробности',
        'который', 'которая', 'которые', 'которое', 'которому', 'которой', 'которыми',
        'https', 'http', 'риа', 'ъузнал'
    ])
    return russian_stopwords.union(custom_stopwords)

# Простая функция для токенизации предложений на русском
def russian_sent_tokenize(text):
    """Простая токенизация предложений для русского языка"""
    # Разделяем по точкам, восклицательным и вопросительным знакам
    sentences = re.split(r'[.!?]+', text)
    # Убираем пустые строки и обрезаем пробелы
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def get_tfidf_corpus(texts, max_features=1000, max_df=0.8, min_df=2):
    """Анализ TF-IDF для всего корпуса текстов"""
    
    # Создаем TF-IDF векторaйзер с русскими стоп-словами
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=max_df,  # игнорировать слова, встречающиеся в более чем 80% документов
        min_df=min_df,  # игнорировать слова, встречающиеся менее чем в 2 документах
        stop_words=list(get_stopwords()),
        ngram_range=(1, 2)  # учитываем униграммы и биграммы
    )
    
    # Обучаем и преобразуем тексты
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    return {
        'vectorizer': tfidf_vectorizer,
        'matrix': tfidf_matrix,
        'feature_names': feature_names
    }

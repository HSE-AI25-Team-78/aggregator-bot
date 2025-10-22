# Быстрый старт

Этот проект позволяет собирать и анализировать тексты из открытых Telegram-каналов.  
Ниже приведены шаги, чтобы быстро запустить всё локально.

---

## 1. Подготовка окружения

1. Клонируй репозиторий и перейди в его директорию:
   ```bash
   git clone <ссылка-на-репозиторий>
   cd telegram-eda-starter
   ```

2. Создай виртуальное окружение и активируй его:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows
   ```

3. Установи зависимости:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Авторизация в Telegram

1. Скопируй пример `.env`:
   ```bash
   cp .env.example .env
   ```

2. Впиши свои данные:
   ```
   API_ID=...
   API_HASH=...
   SESSION_NAME=tg_session
   TG_PHONE=+7XXXXXXXXXX
   TG_2FA=пароль
   ```

3. Пройди авторизацию:
   ```bash
   python test_connection.py
   ```
   После успешного входа появится файл `tg_session.session` — он нужен для работы скриптов.

---

## 3. Парсинг данных

Для примера используется новостной канал [@rbc_news](https://t.me/rbc_news):
```bash
python scripts/fetch.py --channel @rbc_news --limit 2000
```
Результат сохранится в:
```
data/raw/rbc_news.csv
```

---

## 4. Очистка текста

```bash
python scripts/preprocess.py --infile data/raw/rbc_news.csv --outfile data/processed/rbc_news_clean.csv
```
После обработки файл сохранится в:
```
data/processed/rbc_news_clean.csv
```

---

## 5. Разведочный анализ (EDA)

```bash
python scripts/eda.py --infile data/processed/rbc_news_clean.csv --n 30 --out_dir reports/figures
```
После выполнения:
- в консоли появятся топ-слова и биграммы;
- будет создано облако слов `reports/figures/wordcloud.png`.

---

## 6. Что можно сделать дальше
- заменить канал на другой;
- объединить несколько каналов;
- добавить лемматизацию и анализ тональности;
- собрать простой дашборд на Streamlit.

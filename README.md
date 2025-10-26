# Telegram Parser

Простой парсер Telegram-каналов на Python.  
Собирает посты из указанных каналов и сохраняет их в `.csv` для дальнейшей обработки или анализа.

---

## Как устроен проект

```
telegram-parser/
├─ src/
│  ├─ config.py           # настройка путей и загрузка .env
│  ├─ tg_client.py        # подключение к Telegram API через Telethon
│  └─ fetch_messages.py   # функции для получения сообщений и их сохранения
│
├─ scripts/
│  └─ fetch.py            # CLI-скрипт для запуска парсинга
│
├─ data/
│  └─ raw/                # сюда автоматически сохраняются выгруженные CSV-файлы
│
├─ .env.example           # пример конфигурации для подключения к Telegram API
├─ requirements.txt       # зависимости проекта
└─ README.md              # документация (этот файл)
```

---

## Установка

1. Клонируй репозиторий и перейди в его папку:
   ```bash
   git clone https://github.com/HSE-AI25-Team-78/aggregator-bot.git
   cd aggregator-bot
   ```

2. Создай и активируй виртуальное окружение:
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

## Настройка .env

Скопируй пример и заполни его:
```bash
cp .env.example .env
```

Затем впиши свои данные:
```
API_ID=...
API_HASH=...
SESSION_NAME=tg_session
TG_PHONE=+7XXXXXXXXXX
TG_2FA=пароль
```

---

## Использование

Запуск парсинга канала выполняется одной командой:
```bash
python scripts/fetch.py --channel @rbc_news --limit 1000
```

После выполнения данные сохраняются в:
```
data/raw/rbc_news.csv
```

где:
- `@rbc_news` — имя канала, откуда собираются посты;
- `--limit` — количество сообщений для выгрузки.

---

## Что внутри `data/raw`

Все выгруженные CSV-файлы хранятся именно здесь.  
Например:
```
data/raw/
├─ rbc_news.csv
├─ tass_agency.csv
└─ meduzalive.csv
```

Каждая строка содержит:
- дату публикации,
- текст сообщения,
- количество просмотров,
- ссылку (если есть),
- ID поста.

---

## Как это использовать дальше

- Можно объединять несколько выгрузок для анализа в одном файле.  
- Данные подходят для EDA, NLP, построения рекомендательных систем.  
- Этот парсер будет использоваться как модуль в сервисе-агрегаторе.

---

## Пример

```bash
python scripts/fetch.py --channel @tass_agency --limit 500
```

Результат появится в `data/raw/tass_agency.csv`.

---

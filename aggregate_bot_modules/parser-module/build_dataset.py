# parser-module/build_dataset.py

import os
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from parser import Parser  # твой класс из parser.py
from channels_config import CHANNELS


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


async def build_dataset(limit_per_channel: int = 1000):
    # 1. Загружаем конфиг из .env
    load_dotenv()
    api_id = int(os.getenv("API_ID"))
    api_hash = os.getenv("API_HASH")
    session_name = os.getenv("SESSION_NAME", "ml_session")
    phone = os.getenv("TG_PHONE")
    two_fa = os.getenv("TG_2FA")  # может быть None

    # 2. Создаём парсер и коннектимся один раз
    parser = Parser(api_id=api_id, api_hash=api_hash, session_name=session_name)

    print("[*] Подключаемся к Telegram...")
    await parser.connect(phone=phone, two_fa=two_fa)

    all_dfs = []

    # 3. Итерируемся по каналам
    for channel, topic in CHANNELS.items():
        print(f"[*] Забираем сообщения из {channel} (topic={topic})...")

        try:
            df = await parser.fetch(channel=channel, limit=limit_per_channel)
        except Exception as e:
            print(f"[!] Не удалось получить данные из {channel}: {e}")
            continue

        if df.empty:
            print(f"[!] Канал {channel}: нет сообщений, пропускаем")
            continue

        # Добавляем служебные колонки
        df["channel"] = channel
        df["topic"] = topic

        # По желанию: сохранить сырой CSV по каналу
        channel_name = channel.lstrip("@")
        raw_path = DATA_DIR / "raw"
        raw_path.mkdir(exist_ok=True)
        df.to_csv(raw_path / f"{channel_name}.csv", index=False)

        all_dfs.append(df)

    # 4. Склеиваем все в один DataFrame
    if not all_dfs:
        print("[!] Не удалось собрать ни одного канала")
        await parser.disconnect()
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    out_path = DATA_DIR / "raw_posts_labeled.csv"
    full_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[+] Итоговый датасет сохранён: {out_path} (строк: {len(full_df)})")

    await parser.client.disconnect()


if __name__ == "__main__":
    asyncio.run(build_dataset(limit_per_channel=1000))
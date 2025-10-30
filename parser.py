import os
import asyncio
import argparse
from pathlib import Path

import pandas as pd
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from dotenv import load_dotenv, find_dotenv


class Parser:
    """
    Минимальный Telegram-парсер:
    - подключается к API
    - выгружает сообщения из канала
    - сохраняет в CSV внутри ./data
    """

    def __init__(self, api_id: int, api_hash: str, session_name: str = "tg_session"):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)

    async def connect(self, phone: str | None = None, two_fa: str | None = None):
        """
        Подключение к Telegram.
        Если сессия уже сохранена, логин не требуется.
        Иначе — код на телефон + (опционально) пароль 2FA.
        """
        await self.client.connect()
        if await self.client.is_user_authorized():
            return

        if not phone:
            phone = input("Введите номер телефона в формате +7XXXXXXXXXX: ").strip()
        if not phone.startswith("+"):
            phone = "+" + phone

        await self.client.send_code_request(phone)
        code = input("Код из Telegram: ").strip()
        try:
            await self.client.sign_in(phone=phone, code=code)
        except SessionPasswordNeededError:
            if two_fa is None:
                two_fa = input("Пароль двухэтапной проверки (2FA): ").strip()
            await self.client.sign_in(password=two_fa)

    async def fetch(self, channel: str, limit: int = 1000) -> pd.DataFrame:
        """
        Выгружает последние `limit` сообщений из канала в DataFrame.
        """
        rows = []
        async for m in self.client.iter_messages(channel, limit=limit):
            if not m:
                continue
            rows.append({
                "id": m.id,
                "date": m.date.isoformat() if m.date else None,
                "text": m.text or "",
                "views": getattr(m, "views", None),
                "forwards": getattr(m, "forwards", None),
                "replies": (m.replies.replies if m.replies else None),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _default_outpath(channel: str) -> Path:
        # сохраняем ТОЛЬКО в ./data; без создания других директорий
        Path("data").mkdir(exist_ok=True)
        name = channel.strip("@").replace("/", "_")
        return Path("data") / f"{name}.csv"

    def save_csv(self, df: pd.DataFrame, channel: str, out: str | None = None) -> Path:
        out_path = Path(out) if out else self._default_outpath(channel)
        df.to_csv(out_path, index=False)
        return out_path

    async def run(self, channel: str, limit: int, out: str | None, phone: str | None, two_fa: str | None):
        await self.connect(phone=phone, two_fa=two_fa)
        df = await self.fetch(channel=channel, limit=limit)
        path = self.save_csv(df, channel, out)
        print(f"saved: {path} ({len(df)} rows)")
        await self.client.disconnect()


def main():
    # НИЧЕГО не исполняем глобально — только при запуске как скрипт
    load_dotenv(find_dotenv(".env", raise_error_if_not_found=False))

    parser = argparse.ArgumentParser(description="Минимальный Telegram-парсер каналов")
    parser.add_argument("--channel", required=True, help="username/url/id канала (например, @rbc_news)")
    parser.add_argument("--limit", type=int, default=1000, help="кол-во сообщений")
    parser.add_argument("--out", default=None, help="путь к CSV (по умолчанию: ./data/<channel>.csv)")
    parser.add_argument("--phone", default=os.getenv("TG_PHONE"), help="номер телефона (+7...), можно в .env TG_PHONE")
    parser.add_argument("--two_fa", default=os.getenv("TG_2FA"), help="пароль 2FA, можно в .env TG_2FA")
    parser.add_argument("--session", default=os.getenv("SESSION_NAME", "tg_session"), help="имя сессии (по умолчанию tg_session)")
    args = parser.parse_args()

    api_id = int(os.getenv("API_ID", "0"))
    api_hash = os.getenv("API_HASH", "")
    if not api_id or not api_hash:
        raise SystemExit("API_ID / API_HASH не заданы. Добавьте их в .env")

    p = Parser(api_id=api_id, api_hash=api_hash, session_name=args.session)
    asyncio.run(p.run(
        channel=args.channel,
        limit=args.limit,
        out=args.out,
        phone=args.phone,
        two_fa=args.two_fa,
    ))


if __name__ == "__main__":
    main()
from __future__ import annotations

import asyncio
import csv
import os
from pathlib import Path

from telethon import TelegramClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SUMMARY_PATH = PROJECT_ROOT / "ML" / "expanded_corpus_summary.csv"

CHANNELS = [
    "@rbc_news",
    "@forbesrussia",
    "@kommersant",
    "@fincult",
    "@tproger_news",
    "@xakaton_it",
    "@devschacht",
    "@codeblog",
    "@habr_com",
    "@sportsru",
    "@championatrunews",
    "@matchtv",
    "@nplusone",
    "@postnauka",
    "@polden2035",
    "@science_and_life",
    "@rt_russian",
    "@rian_ru",
    "@meduzalive",
    "@bbcrussian",
    "@tass_agency",
    "@mosrutop",
    "@ENews112",
    "@d_code",
    "@headlines_for_traders",
    "@aviadispet4er",
    "@minzdrav_ru",
    "@historyrussi",
    "@kinostro4ka",
    "@karoartcinema",
    "@politica_media",
]


def load_simple_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def serialize_message(message) -> dict[str, object]:
    return {
        "id": message.id,
        "date": message.date.isoformat() if message.date else None,
        "text": message.text or "",
        "views": getattr(message, "views", None),
        "forwards": getattr(message, "forwards", None),
        "replies": (message.replies.replies if message.replies else None),
    }


async def fetch_channel(client: TelegramClient, channel: str, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    async for message in client.iter_messages(channel, limit=limit):
        if not message:
            continue
        rows.append(serialize_message(message))
    return rows


def save_rows(channel: str, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    channel_name = channel.lstrip("@")

    raw_path = RAW_DIR / f"{channel_name}.csv"
    flat_path = DATA_DIR / f"{channel_name}.csv"

    fieldnames = ["id", "date", "text", "views", "forwards", "replies"]
    for out_path in [raw_path, flat_path]:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return raw_path, flat_path


async def main(limit: int = 2500) -> None:
    load_simple_env(PROJECT_ROOT / ".env")
    api_id = int(os.getenv("API_ID", "0"))
    api_hash = os.getenv("API_HASH", "")
    session_name = os.getenv("SESSION_NAME", "tg_session")

    if not api_id or not api_hash:
        raise SystemExit("API_ID/API_HASH not found in .env")

    results: list[dict[str, object]] = []
    client = TelegramClient(str(PROJECT_ROOT / session_name), api_id, api_hash)
    await client.connect()

    if not await client.is_user_authorized():
        raise SystemExit("Telegram session is not authorized. Re-run auth flow first.")

    for channel in CHANNELS:
        try:
            print(f"[*] Fetching {channel} limit={limit}")
            rows = await fetch_channel(client, channel, limit=limit)
            raw_path, flat_path = save_rows(channel, rows)
            results.append(
                {
                    "channel": channel,
                    "rows": len(rows),
                    "status": "ok",
                    "raw_path": str(raw_path),
                    "flat_path": str(flat_path),
                }
            )
            print(f"[+] Saved {len(rows)} rows for {channel}")
        except Exception as exc:
            results.append(
                {
                    "channel": channel,
                    "rows": 0,
                    "status": f"error: {exc}",
                    "raw_path": "",
                    "flat_path": "",
                }
            )
            print(f"[!] Failed {channel}: {exc}")

    await client.disconnect()

    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "rows", "status", "raw_path", "flat_path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[+] Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

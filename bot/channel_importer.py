from __future__ import annotations

import asyncio
import csv
import os
from dataclasses import dataclass
from pathlib import Path

from telethon import TelegramClient


@dataclass(slots=True)
class ImportResult:
    channel: str
    imported_count: int
    status: str
    message: str
    output_path: str | None = None


class ChannelImporter:
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str,
        output_dir: str | Path,
    ) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def import_channel(self, channel: str, limit: int = 200) -> ImportResult:
        return asyncio.run(self._import_channel(channel, limit=limit))

    async def _import_channel(self, channel: str, limit: int = 200) -> ImportResult:
        client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        await client.connect()

        try:
            if not await client.is_user_authorized():
                return ImportResult(
                    channel=channel,
                    imported_count=0,
                    status="auth_required",
                    message=(
                        "Telegram client session is not authorized yet. "
                        "You need to log in once before importing custom channels."
                    ),
                )

            rows: list[dict[str, str | int | None]] = []
            async for message in client.iter_messages(channel, limit=limit):
                text = (message.text or "").strip()
                if not text:
                    continue
                rows.append(
                    {
                        "id": message.id,
                        "date": message.date.isoformat() if message.date else "",
                        "text": text,
                        "views": getattr(message, "views", None),
                        "forwards": getattr(message, "forwards", None),
                        "replies": message.replies.replies if message.replies else None,
                        "channel_short": channel.lstrip("@"),
                    }
                )

            if not rows:
                return ImportResult(
                    channel=channel,
                    imported_count=0,
                    status="empty",
                    message="No text posts were found in this channel.",
                )

            output_path = self.output_dir / f"{channel.lstrip('@')}.csv"
            with output_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["id", "date", "text", "views", "forwards", "replies", "channel_short"],
                )
                writer.writeheader()
                writer.writerows(rows)

            return ImportResult(
                channel=channel,
                imported_count=len(rows),
                status="imported",
                message=f"Imported {len(rows)} posts from {channel}.",
                output_path=str(output_path),
            )
        except Exception as exc:
            return ImportResult(
                channel=channel,
                imported_count=0,
                status="error",
                message=str(exc),
            )
        finally:
            await client.disconnect()


def build_importer(output_dir: str | Path) -> ChannelImporter | None:
    api_id_raw = os.getenv("API_ID")
    api_hash = os.getenv("API_HASH")
    session_name = os.getenv("SESSION_NAME", "tg_session")

    if not api_id_raw or not api_hash:
        return None

    return ChannelImporter(
        api_id=int(api_id_raw),
        api_hash=api_hash,
        session_name=session_name,
        output_dir=output_dir,
    )

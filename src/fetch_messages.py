import pandas as pd
from .tg_client import get_client

async def fetch_channel_messages(channel: str, limit: int = 1000) -> pd.DataFrame:
    msgs = []
    async with get_client() as client:
        async for m in client.iter_messages(channel, limit=limit):
            if not m:
                continue
            msgs.append({
                "id": m.id,
                "date": m.date,
                "text": m.text or "",
                "views": getattr(m, "views", None),
                "forwards": getattr(m, "forwards", None),
                "replies": getattr(m, "replies", None).replies if getattr(m, "replies", None) else None
            })
    return pd.DataFrame(msgs)

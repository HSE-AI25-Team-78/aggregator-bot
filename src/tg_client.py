from telethon import TelegramClient
from .config import API_ID, API_HASH, SESSION_NAME

def get_client() -> TelegramClient:
    return TelegramClient(SESSION_NAME, API_ID, API_HASH)

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    SessionPasswordNeededError,
    AuthRestartError,
    PhonePasswordFloodError,
    PhoneCodeInvalidError,
    PhoneCodeExpiredError,
    PhoneNumberInvalidError,
)
from dotenv import load_dotenv, find_dotenv
import os
import asyncio

# Load .env from project root
load_dotenv(find_dotenv('.env', raise_error_if_not_found=True))

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
SESSION_NAME = os.getenv('SESSION_NAME', 'tg_session')
PHONE = os.getenv('TG_PHONE')
TWO_FA = os.getenv('TG_2FA')   # опционально: пароль 2FA из .env

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

async def send_code_with_retry(client, phone: str, *, max_attempts: int = 6, base_delay: int = 30):

    attempt = 0
    delay = base_delay
    while True:
        attempt += 1
        try:
            print(f' Запрашиваю код (попытка {attempt}/{max_attempts})…')
            return await client.send_code_request(phone)
        except FloodWaitError as e:
            print(f' FloodWait: нужно подождать {e.seconds} сек. Ждём…')
            await asyncio.sleep(e.seconds)
        except PhonePasswordFloodError:
            if attempt >= max_attempts:
                raise
            print(f' Слишком много попыток. Ждём {delay} сек и пробуем снова…')
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15 * 60)

async def main():
    try:
        await client.connect()
        if await client.is_user_authorized():
            me = await client.get_me()
            print(' Уже авторизован как:', me.username or me.id)
            return

        phone = PHONE or input('Введите номер телефона (например, +7XXXXXXXXXX): ').strip()
        if not phone.startswith('+'):
            print('ℹ Добавляю + автоматически.')
            phone = '+' + phone

        try:
            await send_code_with_retry(client, phone)
        except AuthRestartError:
            print(' Telegram просит перезапустить процесс авторизации. Повторите запуск.')
            return
        except PhoneNumberInvalidError:
            print(' Неверный номер телефона. Проверьте формат.')
            return
        except PhonePasswordFloodError:
            print(' Слишком много попыток. Подождите подольше и запустите снова.')
            return

        code = input('Код из Telegram (без пробелов): ').strip()
        try:
            await client.sign_in(phone=phone, code=code)
        except SessionPasswordNeededError:
            pwd = TWO_FA or input('Введите пароль двухэтапной проверки (2FA): ').strip()
            try:
                await client.sign_in(password=pwd)
            except FloodWaitError as e:
                print(f' FloodWait при проверке пароля 2FA: подождите {e.seconds} сек.')
                return
        except PhoneCodeInvalidError:
            print(' Неверный код подтверждения. Запустите ещё раз и введите код внимательно.')
            return
        except PhoneCodeExpiredError:
            print(' Код просрочен. Запустите ещё раз, чтобы запросить новый.')
            return

        me = await client.get_me()
        print(' Успешный вход как:', me.username or me.id)

    finally:
        await client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())

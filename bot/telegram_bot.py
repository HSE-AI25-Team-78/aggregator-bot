from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

from .channel_importer import build_importer
from .recommender import NewsItem, NewsRecommender
from .storage import UserStorage


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT_DIR / "data"
DEFAULT_STORAGE = ROOT_DIR / "bot_data" / "user_profiles.json"
IMPORTED_CHANNELS_DIR = ROOT_DIR / "bot_data" / "imported_channels"
ENV_PATH = ROOT_DIR / ".env"

BACK_BUTTON = "⬅️ Назад"
DONE_BUTTON = "✅ Готово"
SKIP_BUTTON = "⏭ Пропустить"
RESTART_ONBOARDING_BUTTON = "🔄 Настроить заново"
MORE_MENU_BUTTON = "☰ Ещё"

MAIN_MENU_BUTTONS = [
    ["🗞 Дайджест", "📰 Моя лента"],
    ["🔎 Найти новости", MORE_MENU_BUTTON],
]

MORE_MENU_BUTTONS = [
    ["⚙️ Настроить интересы", "🧾 Мой профиль"],
    ["🧹 Сбросить профиль", "ℹ️ Помощь"],
    [BACK_BUTTON],
]

TOPIC_CHOICES = [
    "Общее",
    "Наука и техника",
    "ИТ и телекоммуникации",
    "Общество, государство, политика",
    "Экономика",
    "Медицина",
    "Искусство и культура",
    "Развлечения",
    "Спорт",
    "История",
    "Происшествия",
]
TOPIC_DISPLAY_MAP = {
    "Общее": "Общее",
    "Наука и техника": "Наука и техника",
    "ИТ и телекоммуникации": "IT и телеком",
    "Общество, государство, политика": "Политика и общество",
    "Экономика": "Экономика",
    "Медицина": "Медицина",
    "Искусство и культура": "Культура",
    "Развлечения": "Развлечения",
    "Спорт": "Спорт",
    "История": "История",
    "Происшествия": "Происшествия",
}
DISPLAY_TO_TOPIC_MAP = {value: key for key, value in TOPIC_DISPLAY_MAP.items()}
SOURCE_DISPLAY_MAP = {
    "rbc_news": "РБК",
    "rian_ru": "РИА Новости",
    "tass_agency": "ТАСС",
    "kommersant": "Коммерсант",
    "forbesrussia": "Forbes Russia",
    "habr_com": "Хабр",
    "sportsru": "Sports.ru",
    "bbcrussian": "BBC Russian",
    "rt_russian": "RT Russian",
    "meduzalive": "Meduza",
    "postnauka": "ПостНаука",
    "nplusone": "N + 1",
    "headlines_for_traders": "Headlines for Traders",
}

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MARKUP_ARTIFACT_RE = re.compile(r"[*_`#>{}\[\]]+")

DEFAULT_SOURCE_CHOICES = [
    "kommersant",
    "rbc_news",
    "rian_ru",
    "tass_agency",
    "headlines_for_traders",
    "ENews112",
    "aviadispet4er",
    "d_code",
    "mosrutop",
]
SOURCES_PER_PAGE = 4
MORE_SOURCES_BUTTON = "📚 Показать ещё"
ADD_CHANNEL_BUTTON = "➕ Добавить свой канал"
TOPIC_CONFIDENCE_THRESHOLD = 0.24

LEGACY_TOPIC_MAP = {
    "AI": ["Наука и техника", "ИТ и телекоммуникации"],
    "IT": ["ИТ и телекоммуникации"],
    "Наука": ["Наука и техника"],
    "Политика": ["Общество, государство, политика"],
    "Культура": ["Искусство и культура"],
    "Бизнес": ["Экономика"],
}


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(slots=True)
class EventGroup:
    anchor: NewsItem
    items: list[NewsItem]


class TelegramNewsBot:
    def __init__(self, token: str, recommender: NewsRecommender, storage: UserStorage) -> None:
        self.token = token
        self.recommender = recommender
        self.storage = storage
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.importer = build_importer(IMPORTED_CHANNELS_DIR)

    def _refresh_recommender_if_needed(self) -> None:
        self.recommender.reload_if_changed(min_check_interval_seconds=45)

    def run(self) -> None:
        offset = None
        print("Telegram news bot is running...")
        while True:
            try:
                updates = self._api("getUpdates", {"timeout": 30, "offset": offset})["result"]
                for update in updates:
                    offset = update["update_id"] + 1
                    self._handle_update(update)
            except error.URLError as exc:
                print(f"Network error: {exc}")
                time.sleep(3)
            except Exception as exc:
                print(f"Unexpected error: {exc}")
                time.sleep(1)

    def _handle_update(self, update: dict) -> None:
        if "callback_query" in update:
            self._handle_callback(update["callback_query"])
            return

        message = update.get("message")
        if not message or "text" not in message:
            return

        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        text = message["text"].strip()
        profile = self.storage.get_profile(user_id)

        if text.startswith("/start"):
            self._start_or_restart(chat_id, user_id, profile, force_restart=False)
            return
        if text.startswith("/help"):
            self._send_help(chat_id)
            return
        if text.startswith("/reset"):
            self._reset_profile(chat_id, user_id)
            return
        if text.startswith("/profile"):
            self._send_profile(chat_id, user_id)
            return

        self._handle_text_message(chat_id, user_id, text, profile)

    def _handle_callback(self, callback: dict) -> None:
        self._refresh_recommender_if_needed()
        data = callback.get("data", "")
        user_id = callback["from"]["id"]
        chat_id = callback["message"]["chat"]["id"]
        callback_id = callback["id"]
        profile = self.storage.get_profile(user_id)

        if data.startswith("like:"):
            item_id = data.split(":", 1)[1]
            if item_id not in profile["liked_ids"]:
                profile["liked_ids"].append(item_id)
            if item_id in profile["disliked_ids"]:
                profile["disliked_ids"].remove(item_id)
            self.storage.update_profile(user_id, profile)
            self._api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "Лайк учтён."})
            return

        if data.startswith("dislike:"):
            item_id = data.split(":", 1)[1]
            if item_id not in profile["disliked_ids"]:
                profile["disliked_ids"].append(item_id)
            if item_id in profile["liked_ids"]:
                profile["liked_ids"].remove(item_id)
            self.storage.update_profile(user_id, profile)
            self._api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "Ок, это уберу из приоритета."})
            return

        if data.startswith("more:"):
            item_id = data.split(":", 1)[1]
            self._api("answerCallbackQuery", {"callback_query_id": callback_id, "text": "Подбираю похожие новости..."})
            recommendations = self.recommender.similar_to_item(
                item_id,
                limit=3,
                boosted_sources=self._boosted_sources(profile),
                source_boost=0.12,
                exclude_ids=set(profile["shown_ids"]),
            )
            self._send_news_batch(
                chat_id,
                user_id,
                recommendations,
                "Похожие новости:",
                explanation_mode="similar",
                anchor_item=self.recommender.get_item(item_id),
            )
            return

        self._api("answerCallbackQuery", {"callback_query_id": callback_id})

    def _handle_text_message(self, chat_id: int, user_id: int, text: str, profile: dict) -> None:
        profile = self._sync_profile(user_id, profile)
        if not profile.get("onboarding_completed"):
            self._handle_onboarding_text(chat_id, user_id, text, profile)
            return

        if text == "📰 Моя лента":
            self._send_recommendations(chat_id, user_id)
            return
        if text == "🗞 Дайджест":
            self._send_digest(chat_id, user_id)
            return
        if text == "🔎 Найти новости":
            profile["mode"] = "awaiting_search_query"
            self.storage.update_profile(user_id, profile)
            self._send_message(chat_id, "Напиши тему, событие или ключевые слова.")
            return
        if text == MORE_MENU_BUTTON:
            profile["mode"] = "aux_menu"
            self.storage.update_profile(user_id, profile)
            self._send_message(
                chat_id,
                "Здесь дополнительные действия по профилю и настройкам.",
                reply_markup=self._more_menu_keyboard(),
            )
            return
        if text == "⚙️ Настроить интересы":
            self._start_or_restart(chat_id, user_id, profile, force_restart=True)
            return
        if text == "🧾 Мой профиль":
            self._send_profile(chat_id, user_id)
            return
        if text == "🧹 Сбросить профиль":
            self._reset_profile(chat_id, user_id)
            return
        if text == "ℹ️ Помощь":
            self._send_help(chat_id)
            return
        if text == BACK_BUTTON:
            profile["mode"] = "main"
            self.storage.update_profile(user_id, profile)
            self._send_message(chat_id, "Вернул в главное меню.", reply_markup=self._main_menu_keyboard())
            return

        if profile.get("mode") == "awaiting_search_query":
            profile["mode"] = "main"
            self.storage.update_profile(user_id, profile)
            self._handle_query(chat_id, user_id, text)
            return

        self._handle_query(chat_id, user_id, text)

    def _handle_onboarding_text(self, chat_id: int, user_id: int, text: str, profile: dict) -> None:
        profile = self._sync_profile(user_id, profile)
        stage = profile.get("onboarding_stage", "topics")

        if profile.get("mode") == "awaiting_custom_channel_for_sources":
            self._handle_custom_channel_input(chat_id, user_id, text, profile)
            return

        if text == RESTART_ONBOARDING_BUTTON:
            self._start_or_restart(chat_id, user_id, profile, force_restart=True)
            return

        if stage == "topics":
            self._handle_topics_step(chat_id, user_id, text, profile)
            return
        if stage == "sources":
            self._handle_sources_step(chat_id, user_id, text, profile)
            return

    def _handle_topics_step(self, chat_id: int, user_id: int, text: str, profile: dict) -> None:
        normalized_text = self._normalize_topic_button_text(text)

        if text == DONE_BUTTON:
            if not profile["selected_topics"]:
                self._send_message(chat_id, "Выбери хотя бы одну тему или нажми «⏭ Пропустить».", reply_markup=self._topics_keyboard(profile))
                return
            profile["onboarding_stage"] = "sources"
            self.storage.update_profile(user_id, profile)
            self._send_sources_prompt(chat_id, profile)
            return

        if text == SKIP_BUTTON:
            profile["selected_topics"] = []
            profile["onboarding_stage"] = "sources"
            self.storage.update_profile(user_id, profile)
            self._send_sources_prompt(chat_id, profile)
            return

        if normalized_text not in TOPIC_CHOICES:
            self._send_message(chat_id, "Выбери темы кнопками ниже.", reply_markup=self._topics_keyboard(profile))
            return

        if normalized_text in profile["selected_topics"]:
            profile["selected_topics"].remove(normalized_text)
        else:
            profile["selected_topics"].append(normalized_text)
        self.storage.update_profile(user_id, profile)
        self._send_topics_prompt(chat_id, profile, updated=True)

    def _handle_sources_step(self, chat_id: int, user_id: int, text: str, profile: dict) -> None:
        normalized_text = self._strip_selected_prefix(text)

        if text == DONE_BUTTON:
            self._finish_onboarding(chat_id, user_id, profile)
            return

        if text == SKIP_BUTTON:
            profile["preferred_sources"] = []
            self._finish_onboarding(chat_id, user_id, profile)
            return

        if text == MORE_SOURCES_BUTTON:
            profile["source_page"] = (profile.get("source_page", 0) + 1) % self._source_page_count()
            self.storage.update_profile(user_id, profile)
            self._send_sources_prompt(chat_id, profile, updated=True)
            return

        if text == ADD_CHANNEL_BUTTON:
            profile["mode"] = "awaiting_custom_channel_for_sources"
            self.storage.update_profile(user_id, profile)
            self._send_message(
                chat_id,
                "Отправь username канала в формате `@channel_name`. Можно несколько через пробел.",
                reply_markup=self._custom_channel_input_keyboard(),
            )
            return

        if normalized_text not in self._available_sources():
            self._send_message(chat_id, "Выбирай источники кнопками ниже.", reply_markup=self._sources_keyboard(profile))
            return

        if normalized_text in profile["preferred_sources"]:
            profile["preferred_sources"].remove(normalized_text)
        else:
            profile["preferred_sources"].append(normalized_text)
        self.storage.update_profile(user_id, profile)
        self._send_sources_prompt(chat_id, profile, updated=True)

    def _handle_custom_channel_input(self, chat_id: int, user_id: int, text: str, profile: dict) -> None:
        if text == BACK_BUTTON:
            profile["mode"] = "onboarding"
            self.storage.update_profile(user_id, profile)
            self._send_sources_prompt(chat_id, profile)
            return

        channels = self._extract_channels(text)
        if not channels:
            self._send_message(
                chat_id,
                "Нужен username в формате `@channel_name`.",
                reply_markup=self._custom_channel_input_keyboard(),
            )
            return

        added_sources: list[str] = []
        added_channels: list[str] = []
        import_messages: list[str] = []
        available_sources = set(self._available_sources())

        for channel in channels:
            normalized = channel.lstrip("@")
            if normalized in available_sources:
                if normalized not in profile["preferred_sources"]:
                    profile["preferred_sources"].append(normalized)
                    added_sources.append(normalized)
            else:
                if channel not in profile["custom_channels"]:
                    profile["custom_channels"].append(channel)
                    added_channels.append(channel)

                import_result = self._import_custom_channel(channel)
                profile["custom_channel_status"][channel] = import_result.status
                if import_result.status == "imported":
                    profile["custom_channel_last_imported"][channel] = import_result.imported_count
                import_messages.append(self._format_import_result(import_result))

        profile["mode"] = "onboarding"
        self.storage.update_profile(user_id, profile)

        messages = []
        if added_sources:
            messages.append("Добавил в источники: " + ", ".join(added_sources))
        if added_channels:
            messages.append("Сохранил свои каналы: " + ", ".join(added_channels))
        if import_messages:
            messages.append("Импорт:\n" + "\n".join(import_messages))
        if not messages:
            messages.append("Эти источники уже были добавлены.")

        self._send_sources_prompt(chat_id, profile, updated=True, custom_messages=messages)

    def _start_or_restart(self, chat_id: int, user_id: int, profile: dict, force_restart: bool) -> None:
        if force_restart or not profile.get("onboarding_completed"):
            profile.update(
                {
                    "mode": "onboarding",
                    "onboarding_stage": "topics",
                    "onboarding_completed": False,
                    "selected_topics": [],
                    "preferred_sources": [],
                    "custom_channels": [],
                    "custom_channel_status": {},
                    "custom_channel_last_imported": {},
                    "liked_ids": [],
                    "disliked_ids": [],
                    "shown_ids": [],
                    "last_recommendations": [],
                    "source_page": 0,
                }
            )
            self.storage.update_profile(user_id, profile)
            self._send_message(
                chat_id,
                "Привет! Сначала быстро соберём твой профиль, чтобы лента была персональной.",
                reply_markup=self._topics_keyboard(profile),
            )
            self._send_topics_prompt(chat_id, profile)
            return

        self._send_message(chat_id, "Профиль уже настроен. Можешь пользоваться главным меню.")

    def _send_topics_prompt(self, chat_id: int, profile: dict, updated: bool = False) -> None:
        selected = ", ".join(profile["selected_topics"]) if profile["selected_topics"] else "пока ничего"
        selected = self._format_topics_for_display(profile["selected_topics"]) if profile["selected_topics"] else "пока ничего"
        prefix = "Обновил выбор.\n\n" if updated else ""
        self._send_message(
            chat_id,
            prefix
            + "Шаг 1 из 2. Выбери, что тебе интересно.\n"
            + f"Сейчас выбрано: {selected}\n\n"
            + "Можно выбрать несколько тем, потом нажать «✅ Готово».",
            reply_markup=self._topics_keyboard(profile),
        )

    def _send_sources_prompt(
        self,
        chat_id: int,
        profile: dict,
        updated: bool = False,
        custom_messages: list[str] | None = None,
    ) -> None:
        selected = ", ".join(profile["preferred_sources"]) if profile["preferred_sources"] else "пока ничего"
        channels = ", ".join(profile["custom_channels"]) if profile["custom_channels"] else "пока нет"
        prefix = "Обновил выбор.\n\n" if updated else ""
        page = profile.get("source_page", 0) + 1
        total_pages = self._source_page_count()
        extra = ""
        if custom_messages:
            extra = "\n".join(custom_messages) + "\n\n"
        self._send_message(
            chat_id,
            prefix
            + "Шаг 2 из 2. Выбери любимые новостные источники.\n"
            + f"Сейчас выбрано: {selected}\n\n"
            + f"Свои каналы: {channels}\n\n"
            + f"Страница {page} из {total_pages}.\n\n"
            + extra
            + "Можно выбрать несколько источников, показать ещё или добавить свой канал.",
            reply_markup=self._sources_keyboard(profile),
        )

    def _send_help(self, chat_id: int) -> None:
        self._send_message(
            chat_id,
            "Как работает бот:\n\n"
            "1. При первом запуске мы собираем твой профиль.\n"
            "2. Потом бот строит персональную ленту по интересам, источникам и лайкам.\n"
            "3. Кнопка «🗞 Дайджест» собирает короткую подборку самого важного по твоему профилю.\n"
            "4. Можно искать новости по запросу и добавлять свои каналы в профиль.\n\n"
            "Если Telegram client-сессия уже авторизована, свои каналы импортируются сразу и попадают в корпус рекомендаций.",
        )

    def _send_profile(self, chat_id: int, user_id: int) -> None:
        self._refresh_recommender_if_needed()
        profile = self.storage.get_profile(user_id)
        profile = self._sync_profile(user_id, profile)
        profile = self._refresh_custom_channels(user_id, profile)
        topics = ", ".join(profile["selected_topics"]) if profile["selected_topics"] else "не выбраны"
        topics = self._format_topics_for_display(profile["selected_topics"]) if profile["selected_topics"] else "не выбраны"
        sources = ", ".join(profile["preferred_sources"]) if profile["preferred_sources"] else "не выбраны"
        channels = ", ".join(profile["custom_channels"]) if profile["custom_channels"] else "не добавлены"
        message = (
            f"Онбординг завершён: {'да' if profile['onboarding_completed'] else 'нет'}\n"
            f"Интересы: {topics}\n"
            f"Источники: {sources}\n"
            f"Свои каналы: {channels}\n"
            f"Статусы каналов: {self._format_channel_statuses(profile)}\n"
            f"Лайков: {len(profile['liked_ids'])}\n"
            f"Дизлайков: {len(profile['disliked_ids'])}\n"
            f"Уже показано новостей: {len(profile['shown_ids'])}"
        )
        self._send_message(chat_id, message)

    def _reset_profile(self, chat_id: int, user_id: int) -> None:
        profile = self.storage.build_default_profile()
        self.storage.update_profile(user_id, profile)
        self._send_message(chat_id, "Профиль очищен. Запускаю настройку заново.")
        self._start_or_restart(chat_id, user_id, profile, force_restart=True)

    def _finish_onboarding(self, chat_id: int, user_id: int, profile: dict) -> None:
        profile["onboarding_completed"] = True
        profile["onboarding_stage"] = None
        profile["mode"] = "main"
        self.storage.update_profile(user_id, profile)
        self._send_message(
            chat_id,
            "Профиль собран. Теперь я могу формировать твою персональную ленту.",
            reply_markup=self._main_menu_keyboard(),
        )
        self._send_recommendations(chat_id, user_id)

    def _import_custom_channel(self, channel: str):
        if self.importer is None:
            class SimpleResult:
                status = "config_missing"
                imported_count = 0
                message = "API_ID/API_HASH are missing."
            return SimpleResult()

        result = self.importer.import_channel(channel, limit=200)
        if result.status == "imported":
            self.recommender.reload()
        return result

    def _refresh_custom_channels(self, user_id: int, profile: dict) -> dict:
        needs_save = False
        for channel in profile["custom_channels"]:
            status = profile["custom_channel_status"].get(channel, "pending_import")
            if status == "imported":
                continue

            result = self._import_custom_channel(channel)
            profile["custom_channel_status"][channel] = result.status
            if result.status == "imported":
                profile["custom_channel_last_imported"][channel] = result.imported_count
            needs_save = True

        if needs_save:
            self.storage.update_profile(user_id, profile)
        return profile

    @staticmethod
    def _format_import_result(result) -> str:
        if result.status == "imported":
            return f"{result.channel}: импортировано {result.imported_count} постов."
        if result.status == "auth_required":
            return (
                f"{result.channel}: нужен первый вход в Telegram client API. "
                "После авторизации импорт заработает."
            )
        if result.status == "empty":
            return f"{result.channel}: в канале не нашёл текстовых постов."
        return f"{result.channel}: ошибка импорта ({result.message})."

    @staticmethod
    def _format_channel_statuses(profile: dict) -> str:
        if not profile["custom_channel_status"]:
            return "нет"
        parts = [f"{channel}={status}" for channel, status in profile["custom_channel_status"].items()]
        return ", ".join(parts)

    def _handle_query(self, chat_id: int, user_id: int, text: str) -> None:
        self._refresh_recommender_if_needed()
        profile = self.storage.get_profile(user_id)
        profile = self._sync_profile(user_id, profile)
        profile = self._refresh_custom_channels(user_id, profile)
        recommendations = self.recommender.recommend_for_query(
            text,
            limit=3,
            topics=self._selected_topics_filter(profile),
            min_confidence=TOPIC_CONFIDENCE_THRESHOLD if profile["selected_topics"] else 0.0,
            boosted_sources=self._boosted_sources(profile),
            source_boost=0.15,
            exclude_ids=set(profile["shown_ids"]).union(profile["disliked_ids"]),
        )
        self._send_news_batch(
            chat_id,
            user_id,
            recommendations,
            f"Что нашёл по запросу: {text}",
            explanation_mode="search",
            query_text=text,
        )

    def _send_recommendations(self, chat_id: int, user_id: int) -> None:
        self._refresh_recommender_if_needed()
        profile = self.storage.get_profile(user_id)
        profile = self._sync_profile(user_id, profile)
        profile = self._refresh_custom_channels(user_id, profile)
        recommendations, title, explanation_mode = self._collect_feed_items(profile, limit=3)
        self._send_news_batch(
            chat_id,
            user_id,
            recommendations,
            title,
            explanation_mode=explanation_mode,
        )

    def _send_digest(self, chat_id: int, user_id: int) -> None:
        self._refresh_recommender_if_needed()
        profile = self.storage.get_profile(user_id)
        profile = self._sync_profile(user_id, profile)
        profile = self._refresh_custom_channels(user_id, profile)
        groups, _, explanation_mode = self._collect_feed_items(profile, limit=5)
        if not groups:
            self._send_message(
                chat_id,
                "Сегодня не удалось собрать уверенный дайджест по твоему профилю. "
                "Попробуй расширить интересы или добавить более тематические источники.",
            )
            return
        self._send_message(chat_id, self._build_digest_intro(groups, profile))
        self._send_news_batch(
            chat_id,
            user_id,
            groups,
            "Твой дайджест:",
            explanation_mode=explanation_mode,
        )

    def _collect_feed_items(self, profile: dict, limit: int) -> tuple[list[EventGroup], str, str]:
        topics = self._selected_topics_filter(profile)
        boosted_sources = self._boosted_sources(profile)
        exclude_ids = set(profile["shown_ids"]).union(profile["disliked_ids"])
        candidate_limit = max(limit * 4, limit + 6)

        if profile["liked_ids"]:
            items = self.recommender.recommend_for_profile(
                profile["liked_ids"],
                limit=candidate_limit,
                topics=topics,
                min_confidence=TOPIC_CONFIDENCE_THRESHOLD if topics else 0.0,
                boosted_sources=boosted_sources,
                source_boost=0.12,
                exclude_ids=exclude_ids,
                diversify=False,
            )
            return self._group_feed_items(items, limit), "Персональная лента по твоим интересам:", "profile"

        if topics:
            items = self.recommender.recommend_for_topics(
                limit=candidate_limit,
                topics=topics,
                min_confidence=TOPIC_CONFIDENCE_THRESHOLD,
                boosted_sources=boosted_sources,
                source_boost=0.10,
                exclude_ids=exclude_ids,
                diversify=False,
            )
            return self._group_feed_items(items, limit), "Стартовая лента по выбранным интересам:", "topics"

        items = self.recommender.latest(
            limit=candidate_limit,
            boosted_sources=boosted_sources,
            source_boost=0.08,
            diversify=False,
        )
        return self._group_feed_items(items, limit), "Пока показываю свежие новости:", "latest"

    def _send_news_batch(
        self,
        chat_id: int,
        user_id: int,
        groups: list[EventGroup],
        header: str,
        explanation_mode: str,
        query_text: str | None = None,
        anchor_item: NewsItem | None = None,
    ) -> None:
        if not groups:
            self._send_message(
                chat_id,
                "Пока не нашёл постов, которые модель достаточно уверенно относит к выбранным темам. "
                "Попробуй выбрать другие категории или добавить более тематический источник.",
            )
            return

        self._send_message(chat_id, header)
        profile = self.storage.get_profile(user_id)
        profile = self._sync_profile(user_id, profile)
        last_ids: list[str] = []

        for group in groups:
            item = group.anchor
            last_ids.append(item.item_id)
            if item.item_id not in profile["shown_ids"]:
                profile["shown_ids"].append(item.item_id)
            self._send_message(
                chat_id,
                self._format_news(
                    item,
                    explanation=self._build_explanation(
                        item,
                        profile,
                        mode=explanation_mode,
                        query_text=query_text,
                        anchor_item=anchor_item,
                    ),
                    event_summary=self._build_event_summary(group),
                ),
                reply_markup={
                    "inline_keyboard": [[
                        {"text": "👍 Лайк", "callback_data": f"like:{item.item_id}"},
                        {"text": "👎 Неинтересно", "callback_data": f"dislike:{item.item_id}"},
                        {"text": "🔁 Похожее", "callback_data": f"more:{item.item_id}"},
                    ]]
                },
            )

        profile["last_recommendations"] = last_ids
        self.storage.update_profile(user_id, profile)

    def _send_message(self, chat_id: int, text: str, reply_markup: dict | None = None) -> None:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "reply_markup": reply_markup or self._main_menu_keyboard(),
        }
        self._api("sendMessage", payload)

    def _api(self, method: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error in {method}: {data}")
        return data

    @staticmethod
    def _selected_topics_filter(profile: dict) -> set[str] | None:
        topics = set(profile["selected_topics"])
        return topics or None

    def _boosted_sources(self, profile: dict) -> set[str] | None:
        sources = set()
        for source in profile["preferred_sources"]:
            if source in self.recommender.sources:
                sources.add(source)
        for channel in profile["custom_channels"]:
            normalized = channel.lstrip("@")
            if normalized in self.recommender.sources:
                sources.add(normalized)
        return sources or None

    def _sync_profile(self, user_id: int, profile: dict) -> dict:
        profile = self._normalize_selected_topics(profile)
        valid_ids = set(self.recommender.item_positions)
        changed = False

        for field in ("liked_ids", "disliked_ids", "shown_ids", "last_recommendations"):
            original = profile.get(field, [])
            filtered = [item_id for item_id in original if item_id in valid_ids]
            if field == "shown_ids":
                filtered = filtered[-300:]
            if filtered != original:
                profile[field] = filtered
                changed = True

        max_page = max(0, self._source_page_count() - 1)
        current_page = int(profile.get("source_page", 0))
        bounded_page = min(max(current_page, 0), max_page)
        if bounded_page != current_page:
            profile["source_page"] = bounded_page
            changed = True

        preferred_sources = [source for source in profile["preferred_sources"] if source in self._available_sources()]
        if preferred_sources != profile["preferred_sources"]:
            profile["preferred_sources"] = preferred_sources
            changed = True

        if changed:
            self.storage.update_profile(user_id, profile)
        return profile

    def _build_digest_intro(self, groups: list[EventGroup], profile: dict) -> str:
        items = [group.anchor for group in groups]
        topic_counts = Counter(item.predicted_label for item in items)
        source_counts = Counter(item.source for item in items)
        top_topics = ", ".join(
            TOPIC_DISPLAY_MAP.get(topic, topic)
            for topic, _ in topic_counts.most_common(3)
        )
        top_sources = ", ".join(source for source, _ in source_counts.most_common(3))
        selected_topics = self._format_topics_for_display(profile["selected_topics"]) if profile["selected_topics"] else "без фильтра по темам"
        return (
            "Собрал короткий дайджест по твоему профилю.\n"
            f"Твои темы: {selected_topics}\n"
            f"Что преобладает в подборке: {top_topics or 'смешанная повестка'}\n"
            f"Главные источники в дайджесте: {top_sources or 'разные'}\n"
            f"Уникальных событий в подборке: {len(groups)}"
        )

    def _group_feed_items(self, items: list[NewsItem], limit: int) -> list[EventGroup]:
        groups: list[EventGroup] = []
        for item in items:
            target_group: EventGroup | None = None
            for group in groups:
                if self._same_event(item, group.anchor):
                    target_group = group
                    break
            if target_group is None:
                groups.append(EventGroup(anchor=item, items=[item]))
            else:
                target_group.items.append(item)
            if len(groups) >= limit and len(items) <= limit:
                break
        return groups[:limit]

    def _same_event(self, item: NewsItem, anchor: NewsItem) -> bool:
        if item.item_id == anchor.item_id:
            return True
        if item.predicted_label != anchor.predicted_label:
            return False

        similarity = self.recommender.similarity_between(item.item_id, anchor.item_id)
        if similarity >= 0.76:
            return True

        item_prefix = " ".join(item.clean_text.split()[:14])
        anchor_prefix = " ".join(anchor.clean_text.split()[:14])
        return bool(item_prefix and item_prefix == anchor_prefix)

    @staticmethod
    def _build_event_summary(group: EventGroup) -> str:
        if len(group.items) <= 1:
            return ""
        sources = []
        for item in group.items:
            if item.source not in sources:
                sources.append(item.source)
        related_sources = ", ".join(sources[:4])
        if len(sources) > 4:
            related_sources += ", ..."
        return f"Событие подтверждается ещё в {len(group.items) - 1} публикациях: {related_sources}."

    def _build_explanation(
        self,
        item: NewsItem,
        profile: dict,
        mode: str,
        query_text: str | None = None,
        anchor_item: NewsItem | None = None,
    ) -> str:
        reasons: list[str] = []
        topic_display = TOPIC_DISPLAY_MAP.get(item.predicted_label, item.predicted_label)

        if mode in {"topics", "profile"} and item.predicted_label in profile["selected_topics"]:
            reasons.append(f"тема «{topic_display}»")
        if mode == "profile" and profile["liked_ids"]:
            reasons.append("похоже на твои лайки")
        if mode == "search" and query_text:
            reasons.append(f"подходит под запрос «{query_text}»")
        if mode == "similar" and anchor_item:
            anchor_topic = TOPIC_DISPLAY_MAP.get(anchor_item.predicted_label, anchor_item.predicted_label)
            reasons.append(f"похоже на сюжет по теме «{anchor_topic}»")
        if item.source in profile["preferred_sources"]:
            reasons.append("это твой выбранный источник")
        custom_sources = {channel.lstrip("@") for channel in profile["custom_channels"]}
        if item.source in custom_sources:
            reasons.append("это твой собственный канал")
        if mode == "latest" and not reasons:
            reasons.append("это свежая и уверенно классифицированная новость")
        if not reasons:
            reasons.append("модель уверенно относит её к твоей ленте")
        return "Почему это в ленте: " + "; ".join(reasons) + "."

    @staticmethod
    def _normalize_selected_topics(profile: dict) -> dict:
        normalized: list[str] = []
        for topic in profile.get("selected_topics", []):
            if topic in TOPIC_CHOICES:
                if topic not in normalized:
                    normalized.append(topic)
                continue
            for mapped_topic in LEGACY_TOPIC_MAP.get(topic, []):
                if mapped_topic not in normalized:
                    normalized.append(mapped_topic)
        profile["selected_topics"] = normalized
        return profile

    @staticmethod
    def _format_topics_for_display(topics: list[str]) -> str:
        return ", ".join(TOPIC_DISPLAY_MAP.get(topic, topic) for topic in topics)

    @staticmethod
    def _normalize_topic_button_text(text: str) -> str:
        normalized = text.removeprefix("✅ ").strip()
        return DISPLAY_TO_TOPIC_MAP.get(normalized, normalized)

    @staticmethod
    def _extract_channels(text: str) -> list[str]:
        channels = re.findall(r"@\w+", text)
        return list(dict.fromkeys(channels))

    @staticmethod
    def _strip_selected_prefix(text: str) -> str:
        return text.removeprefix("✅ ").strip()

    @staticmethod
    def _clean_message_text(text: str) -> str:
        normalized = text.replace("\r", "\n")
        normalized = MARKDOWN_LINK_RE.sub(r"\1", normalized)
        normalized = URL_RE.sub(" ", normalized)
        normalized = MARKUP_ARTIFACT_RE.sub(" ", normalized)
        normalized = normalized.replace("|", " ")
        normalized = re.sub(r"\s*\n\s*", "\n", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _extract_primary_link(text: str) -> str | None:
        markdown_links = [match.group(2) for match in MARKDOWN_LINK_RE.finditer(text)]
        direct_links = URL_RE.findall(text)
        all_links = markdown_links + direct_links
        if not all_links:
            return None
        for link in all_links:
            if "t.me/" not in link:
                return link
        return all_links[0]

    @staticmethod
    def _build_headline(clean_text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+", clean_text)
        for part in parts:
            candidate = part.strip(" -\n")
            if len(candidate) >= 30:
                return candidate[:120].rstrip(" .,;:") + ("..." if len(candidate) > 120 else "")
        words = clean_text.split()
        fallback = " ".join(words[:14]).strip()
        return fallback[:120].rstrip(" .,;:") + ("..." if len(words) > 14 else "")

    @staticmethod
    def _build_excerpt(clean_text: str, headline: str) -> str:
        remainder = clean_text
        if headline and remainder.startswith(headline.rstrip(".")):
            remainder = remainder[len(headline.rstrip(".")):].lstrip(" .,:;-")
        remainder = re.sub(r"\s+", " ", remainder).strip()
        if len(remainder) > 320:
            remainder = remainder[:317].rstrip(" .,;:") + "..."
        return remainder

    @staticmethod
    def _format_date_for_card(value: str) -> str:
        try:
            dt = time.strptime(value.replace("T", " ").split("+")[0], "%Y-%m-%d %H:%M:%S")
            return time.strftime("%d.%m %H:%M", dt)
        except ValueError:
            return value[:16] if value else "дата неизвестна"

    @staticmethod
    def _display_source(source: str) -> str:
        return SOURCE_DISPLAY_MAP.get(source, source.replace("_", " ").strip())

    @classmethod
    def _format_news(cls, item: NewsItem, explanation: str, event_summary: str = "") -> str:
        cleaned = cls._clean_message_text(item.text)
        headline = cls._build_headline(cleaned)
        excerpt = cls._build_excerpt(cleaned, headline)
        topic = TOPIC_DISPLAY_MAP.get(item.predicted_label, item.predicted_label)
        source = cls._display_source(item.source)
        date = cls._format_date_for_card(item.published_at)
        link = cls._extract_primary_link(item.text)

        parts = [
            headline or "Новость",
            f"{topic} • {source} • {date}",
            explanation,
        ]
        if event_summary:
            parts.append(event_summary)
        if excerpt:
            parts.append(excerpt)
        if link:
            parts.append(f"Подробнее: {link}")
        return "\n\n".join(parts)

    @staticmethod
    def _main_menu_keyboard() -> dict:
        return {
            "keyboard": [[{"text": text} for text in row] for row in MAIN_MENU_BUTTONS],
            "resize_keyboard": True,
            "is_persistent": True,
        }

    @staticmethod
    def _more_menu_keyboard() -> dict:
        return {
            "keyboard": [[{"text": text} for text in row] for row in MORE_MENU_BUTTONS],
            "resize_keyboard": True,
            "is_persistent": True,
        }

    @staticmethod
    def _topics_keyboard(profile: dict) -> dict:
        rows = []
        for index in range(0, len(TOPIC_CHOICES), 2):
            row = []
            for topic in TOPIC_CHOICES[index:index + 2]:
                prefix = "✅ " if topic in profile["selected_topics"] else ""
                row.append({"text": prefix + TOPIC_DISPLAY_MAP.get(topic, topic)})
            rows.append(row)
        rows.append([{"text": DONE_BUTTON}, {"text": SKIP_BUTTON}])
        return {"keyboard": rows, "resize_keyboard": True, "is_persistent": True}

    def _sources_keyboard(self, profile: dict) -> dict:
        source_choices = self._available_sources()
        current_page = profile.get("source_page", 0)
        start = current_page * SOURCES_PER_PAGE
        end = start + SOURCES_PER_PAGE
        visible_sources = source_choices[start:end]
        rows = []
        for index in range(0, len(visible_sources), 2):
            row = []
            for source in visible_sources[index:index + 2]:
                prefix = "✅ " if source in profile["preferred_sources"] else ""
                row.append({"text": prefix + source})
            rows.append(row)
        rows.append([{"text": MORE_SOURCES_BUTTON}, {"text": ADD_CHANNEL_BUTTON}])
        rows.append([{"text": DONE_BUTTON}, {"text": SKIP_BUTTON}])
        return {"keyboard": rows, "resize_keyboard": True, "is_persistent": True}

    @staticmethod
    def _custom_channel_input_keyboard() -> dict:
        return {
            "keyboard": [[{"text": BACK_BUTTON}]],
            "resize_keyboard": True,
            "is_persistent": True,
        }

    def _available_sources(self) -> list[str]:
        if self.recommender.sources:
            return sorted(self.recommender.sources)
        return sorted(DEFAULT_SOURCE_CHOICES)

    def _source_page_count(self) -> int:
        return max(1, (len(self._available_sources()) + SOURCES_PER_PAGE - 1) // SOURCES_PER_PAGE)


def main() -> None:
    load_local_env(ENV_PATH)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN before running the bot.")

    dataset_path = Path(os.getenv("NEWS_DATASET_PATH", str(DEFAULT_DATASET)))
    storage_path = Path(os.getenv("BOT_STORAGE_PATH", str(DEFAULT_STORAGE)))

    recommender = NewsRecommender(dataset_path, imported_dir=IMPORTED_CHANNELS_DIR)
    storage = UserStorage(storage_path)
    bot = TelegramNewsBot(token=token, recommender=recommender, storage=storage)
    bot.run()


if __name__ == "__main__":
    main()

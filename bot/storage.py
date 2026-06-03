from __future__ import annotations

import json
from pathlib import Path


class UserStorage:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._write({})

    def get_profile(self, user_id: int) -> dict:
        data = self._read()
        key = str(user_id)
        if key not in data:
            data[key] = self.build_default_profile()
            self._write(data)
        else:
            defaults = self.build_default_profile()
            for field, value in defaults.items():
                data[key].setdefault(field, value)
            self._normalize_profile_schema(data[key])
            self._write(data)
        return data[key]

    def update_profile(self, user_id: int, profile: dict) -> None:
        data = self._read()
        self._normalize_profile_schema(profile)
        data[str(user_id)] = profile
        self._write(data)

    def _read(self) -> dict:
        return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_profile_schema(profile: dict) -> None:
        for field in ("liked_ids", "disliked_ids", "shown_ids", "last_recommendations"):
            values = profile.get(field, [])
            normalized: list[str] = []
            for value in values:
                if isinstance(value, str) and ":" in value and value not in normalized:
                    normalized.append(value)
            profile[field] = normalized

        for field in ("preferred_sources", "selected_topics", "custom_channels"):
            values = profile.get(field, [])
            normalized = []
            for value in values:
                if isinstance(value, str) and value and value not in normalized:
                    normalized.append(value)
            profile[field] = normalized

        profile["custom_channel_status"] = {
            str(channel): str(status)
            for channel, status in profile.get("custom_channel_status", {}).items()
            if channel
        }
        profile["custom_channel_last_imported"] = {
            str(channel): int(count)
            for channel, count in profile.get("custom_channel_last_imported", {}).items()
            if channel
        }

    @staticmethod
    def build_default_profile() -> dict:
        return {
            "liked_ids": [],
            "disliked_ids": [],
            "shown_ids": [],
            "preferred_sources": [],
            "selected_topics": [],
            "custom_channels": [],
            "custom_channel_status": {},
            "custom_channel_last_imported": {},
            "last_recommendations": [],
            "mode": "main",
            "onboarding_completed": False,
            "onboarding_stage": "topics",
            "source_page": 0,
        }

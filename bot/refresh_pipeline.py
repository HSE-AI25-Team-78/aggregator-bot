from __future__ import annotations

import argparse
import json
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .channel_importer import ChannelImporter, ImportResult, build_importer
from .telegram_bot import ENV_PATH, IMPORTED_CHANNELS_DIR, ROOT_DIR, DEFAULT_STORAGE, load_local_env
from .storage import UserStorage


HIGH_PRIORITY_CHANNELS = {
    "rbc_news",
    "forbesrussia",
    "kommersant",
    "rian_ru",
    "tass_agency",
    "meduzalive",
    "bbcrussian",
    "rt_russian",
    "headlines_for_traders",
    "ENews112",
    "sportsru",
}
LOW_PRIORITY_CHANNELS = {
    "postnauka",
    "nplusone",
    "historyrussi",
    "karoartcinema",
    "kinostro4ka",
    "minzdrav_ru",
    "codeblog",
}
FAST_NEWS_CHANNELS = {
    "rbc_news",
    "rian_ru",
    "tass_agency",
    "meduzalive",
    "bbcrussian",
    "rt_russian",
    "ENews112",
    "mosrutop",
    "headlines_for_traders",
    "d_code",
    "politica_media",
}
MID_RETENTION_CHANNELS = {
    "forbesrussia",
    "kommersant",
    "prime1",
    "devschacht",
    "codeblog",
    "habr_com",
    "IT_today_ru",
    "sportsru",
    "sportrian",
    "aviadispet4er",
    "madeinrussia_ru",
}
LONG_RETENTION_CHANNELS = {
    "nplusone",
    "postnauka",
    "npnauka",
    "minzdrav_ru",
    "mediamedics",
    "historyrussi",
    "rf_history",
    "kinostro4ka",
    "karoartcinema",
    "Russiacultura",
    "mincultrussia",
}

STATUS_PATH = ROOT_DIR / "bot_data" / "refresh_status.json"
BASE_DATA_DIR = ROOT_DIR / "data"
BASE_CHANNELS_CONFIG = ROOT_DIR / "bot" / "config" / "base_channels.json"


@dataclass(slots=True)
class RefreshEntry:
    channel: str
    scope: str
    due: bool
    stale_minutes: int
    output_path: str
    status: str
    imported_count: int
    message: str
    retention_days: int | None = None
    kept_after_retention: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh news corpus from Telegram channels.")
    parser.add_argument("--force", action="store_true", help="Refresh all channels regardless of staleness.")
    parser.add_argument("--base-limit", type=int, default=1500, help="How many latest posts to keep for base channels.")
    parser.add_argument("--custom-limit", type=int, default=600, help="How many latest posts to keep for user-added channels.")
    parser.add_argument("--max-age-days", type=int, default=30, help="Drop news older than this many days from local CSV files.")
    parser.add_argument(
        "--status-path",
        default=str(STATUS_PATH),
        help="Where to save the refresh summary JSON.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stale_minutes_for(channel: str, scope: str) -> int:
    normalized = channel.lstrip("@")
    if scope == "custom":
        return 60
    if normalized in HIGH_PRIORITY_CHANNELS:
        return 30
    if normalized in LOW_PRIORITY_CHANNELS:
        return 180
    return 90


def retention_days_for(channel: str, scope: str, default_days: int) -> int:
    normalized = channel.lstrip("@")
    if scope == "custom":
        return min(default_days, 30)
    if normalized in FAST_NEWS_CHANNELS:
        return min(default_days, 14)
    if normalized in MID_RETENTION_CHANNELS:
        return min(default_days, 30)
    if normalized in LONG_RETENTION_CHANNELS:
        return max(default_days, 90) if default_days < 90 else default_days
    return default_days


def output_path_for(scope: str, channel: str) -> Path:
    directory = IMPORTED_CHANNELS_DIR if scope == "custom" else BASE_DATA_DIR
    return directory / f"{channel.lstrip('@')}.csv"


def is_due(output_path: Path, stale_minutes: int, force: bool) -> bool:
    if force or not output_path.exists():
        return True
    age_seconds = datetime.now(timezone.utc).timestamp() - output_path.stat().st_mtime
    return age_seconds >= stale_minutes * 60


def collect_base_channels() -> list[str]:
    channels = []
    seen_normalized: set[str] = set()
    if BASE_CHANNELS_CONFIG.exists():
        try:
            payload = json.loads(BASE_CHANNELS_CONFIG.read_text(encoding="utf-8"))
            for channel in payload.get("channels", []):
                normalized = channel.lstrip("@") if isinstance(channel, str) else ""
                if normalized and normalized not in seen_normalized:
                    seen_normalized.add(normalized)
                    channels.append(channel)
        except json.JSONDecodeError:
            pass
    for path in sorted(BASE_DATA_DIR.glob("*.csv")):
        if path.name == "all_channels_combined.csv":
            continue
        stem = path.stem
        normalized = stem.lstrip("@")
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            channels.append(stem)
    return channels


def collect_custom_channels(storage_path: Path) -> list[str]:
    storage = UserStorage(storage_path)
    data = storage._read()  # noqa: SLF001 - controlled local storage format
    channels: list[str] = []
    for profile in data.values():
        for channel in profile.get("custom_channels", []):
            if channel not in channels:
                channels.append(channel)
    return channels


def refresh_channels(
    importer: ChannelImporter | None,
    channels: list[str],
    scope: str,
    limit: int,
    force: bool,
    max_age_days: int,
) -> list[RefreshEntry]:
    entries: list[RefreshEntry] = []
    for raw_channel in channels:
        channel = raw_channel if raw_channel.startswith("@") else f"@{raw_channel}"
        stale_minutes = stale_minutes_for(channel, scope)
        retention_days = retention_days_for(channel, scope, default_days=max_age_days)
        output_path = output_path_for(scope, channel)
        due = is_due(output_path, stale_minutes=stale_minutes, force=force)

        if not importer:
            entries.append(
                RefreshEntry(
                    channel=channel,
                    scope=scope,
                    due=due,
                    stale_minutes=stale_minutes,
                    output_path=str(output_path),
                    status="config_missing",
                    imported_count=0,
                    message="API_ID/API_HASH are missing.",
                    retention_days=retention_days,
                    kept_after_retention=_prune_output_file(output_path, max_age_days=retention_days),
                )
            )
            continue

        if not due:
            kept_count = _prune_output_file(output_path, max_age_days=retention_days)
            entries.append(
                RefreshEntry(
                    channel=channel,
                    scope=scope,
                    due=False,
                    stale_minutes=stale_minutes,
                    output_path=str(output_path),
                    status="skipped",
                    imported_count=0,
                    message="Channel is still fresh enough.",
                    retention_days=retention_days,
                    kept_after_retention=kept_count,
                )
            )
            continue

        result = importer.import_channel(channel, limit=limit)
        entries.append(
            _entry_from_result(
                result,
                scope=scope,
                stale_minutes=stale_minutes,
                output_path=output_path,
                max_age_days=retention_days,
            )
        )
    return entries


def _entry_from_result(
    result: ImportResult,
    scope: str,
    stale_minutes: int,
    output_path: Path,
    max_age_days: int,
) -> RefreshEntry:
    kept_count = _prune_output_file(output_path, max_age_days=max_age_days) if output_path.exists() else None
    return RefreshEntry(
        channel=result.channel,
        scope=scope,
        due=True,
        stale_minutes=stale_minutes,
        output_path=str(output_path),
        status=result.status,
        imported_count=result.imported_count,
        message=result.message,
        retention_days=max_age_days,
        kept_after_retention=kept_count,
    )


def _parse_row_datetime(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _prune_output_file(output_path: Path, max_age_days: int) -> int | None:
    if not output_path.exists():
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    with output_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return 0
        kept_rows = []
        for row in reader:
            row_dt = _parse_row_datetime((row.get("date") or "").strip())
            if row_dt is None or row_dt >= cutoff:
                kept_rows.append(row)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    return len(kept_rows)


def save_status(path: Path, entries: list[RefreshEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at": now_iso(),
        "summary": {
            "total": len(entries),
            "refreshed": sum(1 for entry in entries if entry.status == "imported"),
            "skipped": sum(1 for entry in entries if entry.status == "skipped"),
            "errors": sum(1 for entry in entries if entry.status not in {"imported", "skipped"}),
        },
        "entries": [asdict(entry) for entry in entries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(entries: list[RefreshEntry], status_path: Path) -> None:
    refreshed = [entry for entry in entries if entry.status == "imported"]
    skipped = [entry for entry in entries if entry.status == "skipped"]
    errors = [entry for entry in entries if entry.status not in {"imported", "skipped"}]

    print(f"Refresh completed at {now_iso()}")
    print(f"Refreshed: {len(refreshed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")
    print(f"Status file: {status_path}")

    if refreshed:
        print("Updated channels:")
        for entry in refreshed:
            print(f"- {entry.channel} ({entry.scope}, {entry.imported_count} posts)")

    if errors:
        print("Problems:")
        for entry in errors:
            print(f"- {entry.channel}: {entry.status} ({entry.message})")


def main() -> None:
    args = parse_args()
    load_local_env(ENV_PATH)

    base_importer = build_importer(BASE_DATA_DIR)
    custom_importer = build_importer(IMPORTED_CHANNELS_DIR)
    status_path = Path(args.status_path)
    storage_path = Path(DEFAULT_STORAGE)

    base_channels = collect_base_channels()
    custom_channels = collect_custom_channels(storage_path)

    entries = []
    entries.extend(
        refresh_channels(
            importer=base_importer,
            channels=base_channels,
            scope="base",
            limit=args.base_limit,
            force=args.force,
            max_age_days=args.max_age_days,
        )
    )
    entries.extend(
        refresh_channels(
            importer=custom_importer,
            channels=custom_channels,
            scope="custom",
            limit=args.custom_limit,
            force=args.force,
            max_age_days=args.max_age_days,
        )
    )

    save_status(status_path, entries)
    print_summary(entries, status_path)


if __name__ == "__main__":
    main()

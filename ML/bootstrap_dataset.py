from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_LABELS_PATH = PROJECT_ROOT / "ML" / "raw_posts_labeled.csv"
OUTPUT_DIR = PROJECT_ROOT / "ML" / "bootstrap"


TARGET_LABELS = [
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

CHANNEL_AUTO_LABELS = {
    "habr_com": "ИТ и телекоммуникации",
    "codeblog": "ИТ и телекоммуникации",
    "devschacht": "ИТ и телекоммуникации",
    "nplusone": "Наука и техника",
    "postnauka": "Наука и техника",
    "sportsru": "Спорт",
    "forbesrussia": "Экономика",
    "headlines_for_traders": "Экономика",
    "ENews112": "Происшествия",
    "mosrutop": "Искусство и культура",
    "minzdrav_ru": "Медицина",
    "historyrussi": "История",
    "kinostro4ka": "Развлечения",
    "karoartcinema": "Искусство и культура",
    "politica_media": "Общество, государство, политика",
}

CHANNEL_REVIEW_POOLS = {
    "kommersant",
    "rbc_news",
    "rian_ru",
    "rt_russian",
    "bbcrussian",
    "meduzalive",
    "tass_agency",
    "aviadispet4er",
    "d_code",
}

LABEL_KEYWORDS = {
    "Наука и техника": [
        r"\bучен\w*",
        r"\bархеолог\w*",
        r"\bкосмос\w*",
        r"\bтелескоп\w*",
        r"\bисследован\w*",
        r"\bспутник\w*",
        r"\bфизик\w*",
        r"\bбиолог\w*",
        r"\bнаук\w*",
    ],
    "ИТ и телекоммуникации": [
        r"\biphone\b",
        r"\bapple\b",
        r"\bgoogle\b",
        r"\bmicrosoft\b",
        r"\bopenai\b",
        r"\bchatgpt\b",
        r"\bсмартфон\w*",
        r"\bноутбук\w*",
        r"\bприложени\w*",
        r"\bнейросет\w*",
        r"\bии\b",
        r"\bкибер\w*",
        r"\bхакер\w*",
        r"\bтелеком\w*",
        r"\bютуб\w*",
    ],
    "Общество, государство, политика": [
        r"\bпутин\w*",
        r"\bкремл\w*",
        r"\bпрезидент\w*",
        r"\bминистр\w*",
        r"\bправительств\w*",
        r"\bпарламент\w*",
        r"\bвыбор\w*",
        r"\bсанкц\w*",
        r"\bвойн\w*",
        r"\bукраин\w*",
        r"\bсша\b",
        r"\bес\b",
        r"\bпротест\w*",
    ],
    "Экономика": [
        r"\bакци\w*",
        r"\bинвест\w*",
        r"\bбанк\w*",
        r"\bбизнес\w*",
        r"\bрын\w*",
        r"\bнефт\w*",
        r"\bгаз\w*",
        r"\bдоллар\w*",
        r"\bрубл\w*",
        r"\bвыручк\w*",
        r"\bцен\w*",
        r"\bкомпан\w*",
        r"\bбирж\w*",
    ],
    "Медицина": [
        r"\bврач\w*",
        r"\bболезн\w*",
        r"\bвирус\w*",
        r"\bлечен\w*",
        r"\bпациент\w*",
        r"\bпрепарат\w*",
        r"\bмедицин\w*",
        r"\bздравоохран\w*",
        r"\bвакцин\w*",
    ],
    "Искусство и культура": [
        r"\bмузе\w*",
        r"\bвыставк\w*",
        r"\bтеатр\w*",
        r"\bкино\w*",
        r"\bфильм\w*",
        r"\bсериал\w*",
        r"\bкниг\w*",
        r"\bфестивал\w*",
        r"\bмузык\w*",
        r"\bстуди\w*",
    ],
    "Развлечения": [
        r"\bигр\w*",
        r"\bмем\w*",
        r"\bшоу\b",
        r"\bстрим\w*",
        r"\bюмор\w*",
        r"\bразвлечен\w*",
    ],
    "Спорт": [
        r"\bматч\w*",
        r"\bгол\w*",
        r"\bчемпион\w*",
        r"\bкубок\w*",
        r"\bлига\w*",
        r"\bтренер\w*",
        r"\bсезон\w*",
        r"\bфутбол\w*",
        r"\bхокке\w*",
        r"\bтеннис\w*",
    ],
    "История": [
        r"\bистори\w*",
        r"\bдревн\w*",
        r"\bвек\w*",
        r"\bархив\w*",
        r"\bраскоп\w*",
        r"\bисторическ\w*",
    ],
    "Происшествия": [
        r"\bпогиб\w*",
        r"\bуби\w*",
        r"\bвзрыв\w*",
        r"\bпожар\w*",
        r"\bдтп\b",
        r"\bавари\w*",
        r"\bзадержан\w*",
        r"\bатак\w*",
        r"\bсуд\b",
        r"\bранен\w*",
    ],
}

TEXT_COLUMNS = ("text",)
SOURCE_COLUMNS = ("channel_short", "channel")


@dataclass
class Candidate:
    source_file: str
    channel_short: str
    date: str
    text: str
    clean_text: str
    proposed_label: str
    source_strategy: str
    keyword_hits: str


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|t\.me/\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_existing_clean_texts() -> set[str]:
    existing = set()
    if not RAW_LABELS_PATH.exists():
        return existing
    with RAW_LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            cleaned = clean_text(text)
            if cleaned:
                existing.add(cleaned)
    return existing


def iter_corpus_files() -> list[Path]:
    files = []
    for pattern in ["data/*.csv", "data/raw/*.csv", "bot_data/imported_channels/*.csv"]:
        files.extend(Path(PROJECT_ROOT).glob(pattern))
    return sorted({path.resolve() for path in files if path.is_file() and path.name != "all_channels_combined.csv"})


def get_source(row: dict[str, str], fallback: str) -> str:
    for key in SOURCE_COLUMNS:
        value = (row.get(key) or "").strip()
        if value:
            return value.lstrip("@")
    return fallback


def get_text(row: dict[str, str]) -> str:
    for key in TEXT_COLUMNS:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def keyword_proposals(cleaned_text: str) -> list[str]:
    hits: list[tuple[str, int]] = []
    for label, patterns in LABEL_KEYWORDS.items():
        score = sum(1 for pattern in patterns if re.search(pattern, cleaned_text))
        if score:
            hits.append((label, score))
    hits.sort(key=lambda item: item[1], reverse=True)
    return [label for label, _ in hits]


def build_candidates(max_auto_per_channel: int = 600, max_review_per_channel: int = 600) -> tuple[list[Candidate], list[Candidate]]:
    existing_texts = read_existing_clean_texts()
    auto_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    auto_rows: list[Candidate] = []
    review_rows: list[Candidate] = []

    for path in iter_corpus_files():
        fallback_channel = path.stem
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = get_text(row)
                cleaned = clean_text(text)
                if not cleaned or len(cleaned) < 30:
                    continue
                if cleaned in existing_texts:
                    continue

                channel = get_source(row, fallback_channel)
                date = (row.get("date") or "").strip()

                if channel in CHANNEL_AUTO_LABELS and auto_counts[channel] < max_auto_per_channel:
                    auto_rows.append(
                        Candidate(
                            source_file=path.name,
                            channel_short=channel,
                            date=date,
                            text=text,
                            clean_text=cleaned,
                            proposed_label=CHANNEL_AUTO_LABELS[channel],
                            source_strategy="channel_auto_label",
                            keyword_hits="",
                        )
                    )
                    auto_counts[channel] += 1
                    existing_texts.add(cleaned)
                    continue

                if channel in CHANNEL_REVIEW_POOLS and review_counts[channel] < max_review_per_channel:
                    proposals = keyword_proposals(cleaned)
                    if not proposals:
                        continue
                    review_rows.append(
                        Candidate(
                            source_file=path.name,
                            channel_short=channel,
                            date=date,
                            text=text,
                            clean_text=cleaned,
                            proposed_label=proposals[0],
                            source_strategy="keyword_review",
                            keyword_hits=" | ".join(proposals[:3]),
                        )
                    )
                    review_counts[channel] += 1
                    existing_texts.add(cleaned)

    return auto_rows, review_rows


def write_candidates(path: Path, rows: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "channel_short",
                "date",
                "proposed_label",
                "source_strategy",
                "keyword_hits",
                "text",
                "clean_text",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_file": row.source_file,
                    "channel_short": row.channel_short,
                    "date": row.date,
                    "proposed_label": row.proposed_label,
                    "source_strategy": row.source_strategy,
                    "keyword_hits": row.keyword_hits,
                    "text": row.text,
                    "clean_text": row.clean_text,
                }
            )


def write_summary(path: Path, auto_rows: list[Candidate], review_rows: list[Candidate]) -> None:
    auto_by_label = Counter(row.proposed_label for row in auto_rows)
    review_by_label = Counter(row.proposed_label for row in review_rows)
    auto_by_channel = Counter(row.channel_short for row in auto_rows)
    review_by_channel = Counter(row.channel_short for row in review_rows)

    lines = [
        "# Bootstrap Dataset Summary",
        "",
        f"- Auto candidates: **{len(auto_rows)}**",
        f"- Review candidates: **{len(review_rows)}**",
        "",
        "## Auto by label",
        "",
    ]
    for label, count in auto_by_label.most_common():
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Review by label", ""])
    for label, count in review_by_label.most_common():
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Auto by channel", ""])
    for channel, count in auto_by_channel.most_common():
        lines.append(f"- {channel}: {count}")

    lines.extend(["", "## Review by channel", ""])
    for channel, count in review_by_channel.most_common():
        lines.append(f"- {channel}: {count}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    auto_rows, review_rows = build_candidates()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    auto_path = OUTPUT_DIR / "auto_candidates.csv"
    review_path = OUTPUT_DIR / "review_queue.csv"
    summary_path = OUTPUT_DIR / "summary.md"

    write_candidates(auto_path, auto_rows)
    write_candidates(review_path, review_rows)
    write_summary(summary_path, auto_rows, review_rows)

    print(f"[+] Auto candidates saved to: {auto_path}")
    print(f"[+] Review queue saved to: {review_path}")
    print(f"[+] Summary saved to: {summary_path}")
    print(f"    auto_candidates={len(auto_rows)}")
    print(f"    review_candidates={len(review_rows)}")


if __name__ == "__main__":
    main()

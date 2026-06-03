from __future__ import annotations

import csv
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ML"
BOOTSTRAP_DIR = ML_DIR / "bootstrap"
OUTPUT_DIR = ML_DIR / "pseudo_label_artifacts"

RAW_LABELS_PATH = ML_DIR / "raw_posts_labeled.csv"
AUTO_CANDIDATES_PATH = BOOTSTRAP_DIR / "auto_candidates.csv"
REVIEW_QUEUE_PATH = BOOTSTRAP_DIR / "review_queue.csv"

SERVICE_CONFIG_DIR = PROJECT_ROOT / "service" / "config"
SERVICE_MODELS_DIR = SERVICE_CONFIG_DIR / "models"
REPORT_PATH = OUTPUT_DIR / "report.md"

LABELS = [
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
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

AUTO_ACCEPT_THRESHOLD = 0.53
REVIEW_ACCEPT_THRESHOLD = 0.62
MARGIN_THRESHOLD = 0.10


@dataclass(slots=True)
class TrainedModels:
    vectorizer: TfidfVectorizer
    logreg: LogisticRegression
    nb: MultinomialNB
    label_encoder: LabelEncoder
    train_matrix: object
    train_labels: np.ndarray


def clean_text(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|t\.me/\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_seed_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 15].copy()
    df["label_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    return df


def load_candidates() -> pd.DataFrame:
    auto_df = pd.read_csv(AUTO_CANDIDATES_PATH)
    review_df = pd.read_csv(REVIEW_QUEUE_PATH)
    auto_df["candidate_kind"] = "auto"
    review_df["candidate_kind"] = "review"
    df = pd.concat([auto_df, review_df], ignore_index=True)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)
    return df


def build_models(train_df: pd.DataFrame) -> TrainedModels:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=15000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["clean_text"].astype(str))

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label_name"])

    logreg = LogisticRegression(
        max_iter=4000,
        C=2.0,
        class_weight="balanced",
    )
    logreg.fit(X_train, y_train)

    nb = MultinomialNB(alpha=0.7)
    nb.fit(X_train, y_train)

    return TrainedModels(
        vectorizer=vectorizer,
        logreg=logreg,
        nb=nb,
        label_encoder=le,
        train_matrix=X_train,
        train_labels=y_train,
    )


def keyword_prior(keyword_hits: str) -> np.ndarray:
    scores = np.zeros(len(LABELS), dtype=float)
    if not isinstance(keyword_hits, str) or not keyword_hits.strip():
        return scores
    labels = [part.strip() for part in keyword_hits.split("|") if part.strip()]
    weights = [1.0, 0.6, 0.3]
    for idx, label in enumerate(labels[:3]):
        if label in LABEL_TO_INDEX:
            scores[LABEL_TO_INDEX[label]] += weights[idx]
    if scores.sum() > 0:
        scores /= scores.sum()
    return scores


def proposed_label_prior(label: str, candidate_kind: str) -> np.ndarray:
    scores = np.zeros(len(LABELS), dtype=float)
    if label in LABEL_TO_INDEX:
        scores[LABEL_TO_INDEX[label]] = 1.0
    if scores.sum() > 0:
        scores /= scores.sum()
    return scores


def nearest_neighbor_prior(models: TrainedModels, features, top_k: int = 5) -> np.ndarray:
    similarities = models.train_matrix @ features.T
    scores = np.asarray(similarities.toarray()).reshape(-1)
    if scores.size == 0:
        return np.zeros(len(LABELS), dtype=float)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    prior = np.zeros(len(LABELS), dtype=float)
    for index in top_indices:
        sim = float(scores[index])
        if sim <= 0:
            continue
        label_idx = int(models.train_labels[index])
        prior[label_idx] += sim
    if prior.sum() > 0:
        prior /= prior.sum()
    return prior


def channel_hint(channel_short: str) -> np.ndarray:
    scores = np.zeros(len(LABELS), dtype=float)
    mapping = {
        "habr_com": "ИТ и телекоммуникации",
        "codeblog": "ИТ и телекоммуникации",
        "devschacht": "ИТ и телекоммуникации",
        "d_code": "ИТ и телекоммуникации",
        "nplusone": "Наука и техника",
        "postnauka": "Наука и техника",
        "sportsru": "Спорт",
        "forbesrussia": "Экономика",
        "headlines_for_traders": "Экономика",
        "ENews112": "Происшествия",
        "mosrutop": "Искусство и культура",
        "aviadispet4er": "Наука и техника",
    }
    label = mapping.get(str(channel_short))
    if label:
        scores[LABEL_TO_INDEX[label]] = 1.0
    return scores


def combine_scores(
    models: TrainedModels,
    row: pd.Series,
) -> tuple[str, float, float, dict[str, float]]:
    features = models.vectorizer.transform([row["clean_text"]])

    logreg_proba = models.logreg.predict_proba(features)[0]
    nb_proba = models.nb.predict_proba(features)[0]
    nn_proba = nearest_neighbor_prior(models, features)
    keyword_proba = keyword_prior(str(row.get("keyword_hits", "")))
    proposed_proba = proposed_label_prior(str(row.get("proposed_label", "")), str(row.get("candidate_kind", "")))
    channel_proba = channel_hint(str(row.get("channel_short", "")))

    if str(row.get("candidate_kind")) == "auto":
        combined = (
            0.22 * logreg_proba
            + 0.13 * nb_proba
            + 0.15 * nn_proba
            + 0.35 * proposed_proba
            + 0.15 * channel_proba
        )
    else:
        combined = (
            0.28 * logreg_proba
            + 0.18 * nb_proba
            + 0.22 * nn_proba
            + 0.22 * keyword_proba
            + 0.10 * channel_proba
        )

    total = combined.sum()
    if total <= 0:
        combined = logreg_proba
        total = combined.sum()
    combined = combined / total

    best_idx = int(np.argmax(combined))
    sorted_scores = np.sort(combined)[::-1]
    top_conf = float(sorted_scores[0])
    margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

    debug = {
        "logreg_top": LABELS[int(np.argmax(logreg_proba))],
        "nb_top": LABELS[int(np.argmax(nb_proba))],
        "nn_top": LABELS[int(np.argmax(nn_proba))] if nn_proba.sum() else "",
        "combined_top": LABELS[best_idx],
    }
    return LABELS[best_idx], top_conf, margin, debug


def select_pseudolabels(models: TrainedModels, candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_rows = []
    rejected_rows = []

    for _, row in candidate_df.iterrows():
        final_label, confidence, margin, debug = combine_scores(models, row)
        candidate_kind = str(row["candidate_kind"])
        proposed_label = str(row["proposed_label"])

        if candidate_kind == "auto":
            accepted = confidence >= AUTO_ACCEPT_THRESHOLD or final_label == proposed_label
        else:
            accepted = confidence >= REVIEW_ACCEPT_THRESHOLD and margin >= MARGIN_THRESHOLD

        out_row = {
            **row.to_dict(),
            "final_label": final_label,
            "final_topic_id": LABEL_TO_INDEX[final_label],
            "confidence": round(confidence, 6),
            "margin": round(margin, 6),
            "logreg_top": debug["logreg_top"],
            "nb_top": debug["nb_top"],
            "nn_top": debug["nn_top"],
        }

        if accepted:
            accepted_rows.append(out_row)
        else:
            rejected_rows.append(out_row)

    return pd.DataFrame(accepted_rows), pd.DataFrame(rejected_rows)


def evaluate_model(
    vectorizer: TfidfVectorizer,
    model,
    label_encoder: LabelEncoder,
    test_df: pd.DataFrame,
) -> tuple[float, str]:
    X_test = vectorizer.transform(test_df["clean_text"].astype(str))
    y_test = label_encoder.transform(test_df["label_name"])
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(label) for label in label_encoder.classes_],
        zero_division=0,
    )
    return f1, report


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def read_previous_best_nb_f1() -> float:
    if not REPORT_PATH.exists():
        return 0.0
    text = REPORT_PATH.read_text(encoding="utf-8")
    marker = "Augmented MultinomialNB F1_macro: **"
    for line in text.splitlines():
        if marker in line:
            value = line.split(marker, 1)[1].split("**", 1)[0]
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_dataset()
    candidate_df = load_candidates()
    previous_best_nb_f1 = read_previous_best_nb_f1()

    train_df, test_df = train_test_split(
        seed_df,
        test_size=0.15,
        stratify=seed_df["label_name"],
        random_state=42,
    )

    baseline_models = build_models(train_df)
    baseline_logreg_f1, baseline_logreg_report = evaluate_model(
        baseline_models.vectorizer, baseline_models.logreg, baseline_models.label_encoder, test_df
    )
    baseline_nb_f1, baseline_nb_report = evaluate_model(
        baseline_models.vectorizer, baseline_models.nb, baseline_models.label_encoder, test_df
    )

    accepted_df, rejected_df = select_pseudolabels(baseline_models, candidate_df)

    augmented_train = pd.concat(
        [
            train_df[["channel_short", "date", "topic", "text", "clean_text", "label_name"]],
            accepted_df.rename(columns={"final_topic_id": "topic", "final_label": "label_name"})[
                ["channel_short", "date", "topic", "text", "clean_text", "label_name"]
            ],
        ],
        ignore_index=True,
    )

    final_models = build_models(augmented_train)
    final_logreg_f1, final_logreg_report = evaluate_model(
        final_models.vectorizer, final_models.logreg, final_models.label_encoder, test_df
    )
    final_nb_f1, final_nb_report = evaluate_model(
        final_models.vectorizer, final_models.nb, final_models.label_encoder, test_df
    )

    all_labeled_df = pd.concat([accepted_df, rejected_df], ignore_index=True)

    accepted_df.to_csv(OUTPUT_DIR / "pseudo_labeled_selected.csv", index=False, encoding="utf-8")
    rejected_df.to_csv(OUTPUT_DIR / "pseudo_labeled_rejected.csv", index=False, encoding="utf-8")
    all_labeled_df.to_csv(OUTPUT_DIR / "pseudo_labeled_all.csv", index=False, encoding="utf-8")
    augmented_train.to_csv(OUTPUT_DIR / "augmented_train_dataset.csv", index=False, encoding="utf-8")

    improved = final_nb_f1 >= max(baseline_nb_f1, previous_best_nb_f1)
    if improved:
        save_pickle(SERVICE_CONFIG_DIR / "vectorizer.pkl", final_models.vectorizer)
        save_pickle(SERVICE_CONFIG_DIR / "label_encoder.pkl", final_models.label_encoder)
        save_pickle(SERVICE_MODELS_DIR / "MultinomialNB.pkl", final_models.nb)

    report_lines = [
        "# Pseudo Label Pipeline Report",
        "",
        f"- Seed labeled rows: **{len(seed_df)}**",
        f"- External candidates: **{len(candidate_df)}**",
        f"- Accepted pseudo labels: **{len(accepted_df)}**",
        f"- Rejected candidates: **{len(rejected_df)}**",
        f"- Augmented train rows: **{len(augmented_train)}**",
        "",
        "## Accepted by label",
        "",
    ]
    for label, count in Counter(accepted_df["final_label"]).most_common():
        report_lines.append(f"- {label}: {count}")

    report_lines.extend(
        [
            "",
            "## Metrics on original labeled holdout",
            "",
            f"- Baseline LogisticRegression F1_macro: **{baseline_logreg_f1:.4f}**",
            f"- Baseline MultinomialNB F1_macro: **{baseline_nb_f1:.4f}**",
            f"- Previous best exported MultinomialNB F1_macro: **{previous_best_nb_f1:.4f}**",
            f"- Augmented LogisticRegression F1_macro: **{final_logreg_f1:.4f}**",
            f"- Augmented MultinomialNB F1_macro: **{final_nb_f1:.4f}**",
            f"- Service export updated: **{'yes' if improved else 'no'}**",
            "",
            "## Baseline NB report",
            "",
            "```text",
            baseline_nb_report,
            "```",
            "",
            "## Augmented NB report",
            "",
            "```text",
            final_nb_report,
            "```",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[+] Selected pseudo labels: {len(accepted_df)}")
    print(f"[+] Rejected pseudo labels: {len(rejected_df)}")
    print(f"[+] Baseline NB F1_macro: {baseline_nb_f1:.4f}")
    print(f"[+] Augmented NB F1_macro: {final_nb_f1:.4f}")
    print(f"[+] Service export updated: {'yes' if improved else 'no'}")


if __name__ == "__main__":
    main()

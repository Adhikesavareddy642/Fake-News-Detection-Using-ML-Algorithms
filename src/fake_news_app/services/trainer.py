from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fake_news_app.config import MODEL_PATH

KNOWN_LABELS = {
    "FAKE": 0,
    "REAL": 1,
    "FALSE": 0,
    "TRUE": 1,
    "0": 0,
    "1": 1,
}

REVERSE_LABEL_MAP = {0: "FAKE", 1: "REAL"}


def _map_labels(raw_labels: list[str]) -> list[int]:
    mapped: list[int | None] = []
    for label in raw_labels:
        key = str(label).strip().upper()
        mapped.append(KNOWN_LABELS.get(key))

    if all(value in (0, 1) for value in mapped):
        return [int(value) for value in mapped]

    normalized = [str(label).strip() for label in raw_labels]
    unique = list(dict.fromkeys(normalized))
    if len(unique) != 2:
        raise ValueError(
            "Label column must contain binary classes (2 unique values) or known labels (FAKE/REAL, TRUE/FALSE, 0/1)."
        )
    label_map = {unique[0]: 0, unique[1]: 1}
    return [label_map[value] for value in normalized]


def _read_dataset(csv_path: Path, text_column: str, label_column: str) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    raw_labels: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        if text_column not in headers:
            raise ValueError(f"Text column '{text_column}' not found. Available columns: {sorted(headers)}")
        if label_column not in headers:
            raise ValueError(f"Label column '{label_column}' not found. Available columns: {sorted(headers)}")

        for row in reader:
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not text or not label:
                continue
            texts.append(text)
            raw_labels.append(label)

    if not texts:
        raise ValueError("No valid rows found after filtering empty text/label values.")

    labels = _map_labels(raw_labels)
    return texts, labels


def train_and_export_from_csv(
    csv_path: Path,
    text_column: str = "text",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    texts, labels = _read_dataset(csv_path, text_column=text_column, label_column=label_column)

    class_counts = Counter(labels)
    if min(class_counts.values()) < 2:
        raise ValueError("Each class must have at least 2 samples for train/validation split.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    accuracy = float(pipeline.score(X_test, y_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "accuracy": accuracy,
        "total_samples": len(texts),
        "train_samples": len(X_train),
        "validation_samples": len(X_test),
        "class_distribution": dict(class_counts),
    }


def train_and_export_from_lists(
    texts: list[str],
    labels: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if not texts:
        raise ValueError("No training samples provided.")

    cleaned_texts: list[str] = []
    cleaned_labels: list[int] = []
    for text, label in zip(texts, labels):
        value = (text or "").strip()
        if not value:
            continue
        cleaned_texts.append(value)
        cleaned_labels.append(int(label))

    if not cleaned_texts:
        raise ValueError("No non-empty training samples provided.")

    class_counts = Counter(cleaned_labels)
    if set(class_counts.keys()) != {0, 1}:
        raise ValueError("Training data must include both classes: fake and real.")
    if min(class_counts.values()) < 2:
        raise ValueError("Each class must have at least 2 samples for train/validation split.")

    total_samples = len(cleaned_texts)
    class_count = len(class_counts)
    validation_samples = math.ceil(test_size * total_samples)
    train_samples = total_samples - validation_samples
    if validation_samples < class_count or train_samples < class_count:
        raise ValueError(
            "Invalid split for binary classification. Increase samples or use a test_size such as 0.2 to 0.4."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        cleaned_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=cleaned_labels,
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    accuracy = float(pipeline.score(X_test, y_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "accuracy": accuracy,
        "total_samples": total_samples,
        "train_samples": len(X_train),
        "validation_samples": len(X_test),
        "class_distribution": dict(class_counts),
    }


def append_samples_to_csv(
    csv_path: Path,
    texts: list[str],
    labels: list[int],
    text_column: str = "text",
    label_column: str = "label",
) -> int:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    rows_written = 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[text_column, label_column])
        if not file_exists:
            writer.writeheader()
        for text, label in zip(texts, labels):
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            normalized_label = int(label)
            if normalized_label not in REVERSE_LABEL_MAP:
                raise ValueError("Labels must be 0 (fake) or 1 (real).")
            writer.writerow({text_column: cleaned, label_column: REVERSE_LABEL_MAP[normalized_label]})
            rows_written += 1

    return rows_written


def load_training_data_from_csv(
    csv_path: Path,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[list[str], list[int]]:
    return _read_dataset(csv_path=csv_path, text_column=text_column, label_column=label_column)

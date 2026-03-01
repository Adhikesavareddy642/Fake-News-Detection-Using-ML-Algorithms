from __future__ import annotations

import csv
import json
from collections import Counter
from io import StringIO

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fake_news_app.config import CUSTOM_SAMPLES_PATH, DATASET_PATH, MODEL_PATH

VALID_TEXT_COLUMNS = ('text', 'content', 'article', 'news')
VALID_LABEL_COLUMNS = ('label', 'class', 'target')


def _normalize_label(raw):
    value = str(raw or '').strip().upper()
    if value in {'FAKE', '0'}:
        return 0
    if value in {'REAL', '1'}:
        return 1
    return None


def normalize_text(text: str):
    return ' '.join(str(text or '').split()).strip().lower()


def _resolve_column(fieldnames, candidates):
    if not fieldnames:
        return None
    lookup = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def parse_rows(rows):
    rows = list(rows)
    if not rows:
        raise RuntimeError('No rows found in dataset.')

    text_col = _resolve_column(rows[0].keys(), VALID_TEXT_COLUMNS)
    label_col = _resolve_column(rows[0].keys(), VALID_LABEL_COLUMNS)
    if not text_col or not label_col:
        raise RuntimeError(
            "Dataset must include text and label columns. Accepted headers: "
            f"text={VALID_TEXT_COLUMNS}, label={VALID_LABEL_COLUMNS}."
        )

    texts: list[str] = []
    labels: list[int] = []
    for row in rows:
        text = str(row.get(text_col) or '').strip()
        label = _normalize_label(row.get(label_col))
        if not text or label is None:
            continue
        texts.append(text)
        labels.append(label)

    if not texts:
        raise RuntimeError(
            'No valid training rows found. Each row needs non-empty text and label FAKE/REAL (or 0/1).'
        )
    return texts, labels


def parse_csv_text(csv_text: str):
    reader = csv.DictReader(StringIO(csv_text))
    return parse_rows(reader)


def parse_csv_file(file_storage):
    content = file_storage.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8-sig', errors='replace')
    return parse_csv_text(content)


def load_default_dataset():
    if not DATASET_PATH.exists():
        raise RuntimeError(f'Dataset file not found at {DATASET_PATH}.')
    with DATASET_PATH.open(newline='', encoding='utf-8') as f:
        return parse_rows(csv.DictReader(f))


def train_and_export_model(texts, labels):
    sample_count = len(texts)
    if sample_count < 4:
        raise RuntimeError('Need at least 4 valid rows to train a model.')

    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise RuntimeError('Need both classes (FAKE and REAL) in training data.')

    test_count = max(1, int(sample_count * 0.2))
    if test_count >= sample_count:
        test_count = sample_count - 1

    can_stratify = min(label_counts.values()) >= 2 and test_count >= len(label_counts)
    stratify = labels if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_count,
        random_state=42,
        stratify=stratify,
    )

    pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
            ('clf', LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    accuracy = float(pipeline.score(X_test, y_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return {
        'accuracy': accuracy,
        'samples': sample_count,
        'model_path': str(MODEL_PATH),
    }


def save_custom_samples(texts, labels):
    """Persist custom form samples for exact-match override at inference time."""
    mapping = {}
    if CUSTOM_SAMPLES_PATH.exists():
        try:
            with CUSTOM_SAMPLES_PATH.open(encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, list):
                for item in existing:
                    key = normalize_text(item.get('text', ''))
                    label = _normalize_label(item.get('label'))
                    if key and label is not None:
                        mapping[key] = {'text': item.get('text', ''), 'label': int(label)}
        except Exception:
            mapping = {}

    for text, label in zip(texts, labels):
        key = normalize_text(text)
        if key:
            mapping[key] = {'text': text, 'label': int(label)}

    CUSTOM_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CUSTOM_SAMPLES_PATH.open('w', encoding='utf-8') as f:
        json.dump(list(mapping.values()), f, ensure_ascii=True, indent=2)

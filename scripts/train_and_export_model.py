from __future__ import annotations

import csv
import sys
from pathlib import Path
from urllib.request import urlopen

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_app.config import DATASET_PATH, MODEL_PATH

DATA_URL = 'https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv'


def parse_rows(rows):
    texts: list[str] = []
    labels: list[int] = []

    for row in rows:
        text = (row.get('text') or '').strip()
        label = (row.get('label') or '').strip().upper()
        if not text or label not in {'FAKE', 'REAL'}:
            continue
        texts.append(text)
        labels.append(0 if label == 'FAKE' else 1)

    if not texts:
        raise RuntimeError('No training rows were loaded from dataset.')

    return texts, labels


def load_dataset():
    if DATASET_PATH.exists():
        with DATASET_PATH.open(newline='', encoding='utf-8') as f:
            return parse_rows(csv.DictReader(f))

    with urlopen(DATA_URL) as resp:
        rows = csv.DictReader((line.decode('utf-8') for line in resp))
        return parse_rows(rows)


def train_and_export_model():
    texts, labels = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
            ('clf', LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)

    acc = pipeline.score(X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f'Saved model to: {MODEL_PATH}')
    print(f'Validation accuracy: {acc:.4f}')


def main():
    train_and_export_model()


if __name__ == '__main__':
    main()

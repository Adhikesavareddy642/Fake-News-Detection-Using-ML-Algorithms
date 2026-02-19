from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_app.config import DATASET_PATH, MODEL_PATH
from fake_news_app.services.trainer import train_and_export_from_csv

DATA_URL = 'https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv'


def load_dataset():
    if DATASET_PATH.exists():
        return

    with urlopen(DATA_URL) as resp:
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATASET_PATH.open("w", encoding="utf-8", newline="") as out:
            out.write(resp.read().decode("utf-8"))


def train_and_export_model():
    load_dataset()
    metrics = train_and_export_from_csv(
        csv_path=DATASET_PATH,
        text_column="text",
        label_column="label",
        test_size=0.2,
        random_state=42,
    )
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")


def main():
    train_and_export_model()


if __name__ == '__main__':
    main()

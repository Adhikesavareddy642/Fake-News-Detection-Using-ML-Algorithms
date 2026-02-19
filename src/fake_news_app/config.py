from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
DATASETS_DIR = BASE_DIR / 'datasets'
MODEL_PATH = ARTIFACTS_DIR / 'fake_news_pipeline.joblib'
DATASET_PATH = DATASETS_DIR / 'fake_or_real_news.csv'
FORM_DATASET_PATH = DATASETS_DIR / "form_training_data.csv"
FLASK_DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'

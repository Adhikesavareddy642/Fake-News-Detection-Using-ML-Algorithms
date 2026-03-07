from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / '.env')


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or '').split(',') if item.strip())


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: str, default: bool = True) -> bool:
    raw = str(value or '').strip().lower()
    if raw in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def _unique_items(*groups: tuple[str, ...]) -> tuple[str, ...]:
    ordered = {}
    for group in groups:
        for item in group:
            key = item.strip()
            if key:
                ordered[key] = None
    return tuple(ordered.keys())


ARTIFACTS_DIR = BASE_DIR / 'artifacts'
DATASETS_DIR = BASE_DIR / 'datasets'
MODEL_PATH = ARTIFACTS_DIR / 'fake_news_pipeline.joblib'
CUSTOM_SAMPLES_PATH = ARTIFACTS_DIR / 'custom_samples.json'
DATASET_PATH = DATASETS_DIR / 'fake_or_real_news.csv'
GNEWS_API_URL = os.getenv('GNEWS_API_URL', 'https://gnews.io/api/v4/search')
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY', '').strip()
GNEWS_TIMEOUT_SECONDS = int(os.getenv('GNEWS_TIMEOUT_SECONDS', '12'))
GNEWS_COUNTRY = os.getenv('GNEWS_COUNTRY', 'in').strip().lower()
PREFER_INDIA_NEWS = _parse_bool(os.getenv('PREFER_INDIA_NEWS', '1'))
GNEWS_QUERY_INDIA_TERMS_LIMIT = min(20, max(0, _parse_int(os.getenv('GNEWS_QUERY_INDIA_TERMS_LIMIT', '10'), 10)))
GNEWS_INCLUDE_TRUSTED_IN_QUERY = _parse_bool(os.getenv('GNEWS_INCLUDE_TRUSTED_IN_QUERY', '1'))
GNEWS_QUERY_TRUSTED_TERMS_LIMIT = min(24, max(0, _parse_int(os.getenv('GNEWS_QUERY_TRUSTED_TERMS_LIMIT', '12'), 12)))
DEFAULT_PREFERRED_INDIA_NEWS_SOURCES = (
    'thehindu.com',
    'indiatimes.com',
    'hindustantimes.com',
    'indianexpress.com',
    'ndtv.com',
    'deccanherald.com',
    'livemint.com',
    'moneycontrol.com',
    'business-standard.com',
    'theprint.in',
    'scroll.in',
    'aninews.in',
    'newindianexpress.com',
    'indiatoday.in',
    'outlookindia.com',
    'economictimes.indiatimes.com',
    'financialexpress.com',
    'news18.com',
    'timesnownews.com',
    'wionews.com',
    'theweek.in',
    'telegraphindia.com',
    'tribuneindia.com',
    'thestatesman.com',
    'pib.gov.in',
)
DEFAULT_TRUSTED_NEWS_SOURCES = (
    'reuters',
    'reuters.com',
    'associated press',
    'apnews.com',
    'bbc',
    'bbc.com',
    'bbc.co.uk',
    'the new york times',
    'nytimes.com',
    'the washington post',
    'washingtonpost.com',
    'wall street journal',
    'wsj.com',
    'the guardian',
    'theguardian.com',
    'financial times',
    'ft.com',
    'the economist',
    'economist.com',
    'npr',
    'npr.org',
    'pbs news',
    'pbs.org',
    'bloomberg',
    'bloomberg.com',
    'forbes',
    'forbes.com',
    'cnn',
    'cnn.com',
    'fox news',
    'foxnews.com',
    'abc news',
    'abcnews.go.com',
    'cbs news',
    'cbsnews.com',
    'nbc news',
    'nbcnews.com',
    'msnbc',
    'msnbc.com',
    'usa today',
    'usatoday.com',
    'los angeles times',
    'latimes.com',
    'newsweek',
    'newsweek.com',
    'time',
    'time.com',
    'politico',
    'politico.com',
    'axios',
    'axios.com',
    'the hill',
    'thehill.com',
    'business insider',
    'businessinsider.com',
    'marketwatch',
    'marketwatch.com',
    'al jazeera',
    'aljazeera.com',
    'dw',
    'dw.com',
    'france 24',
    'france24.com',
    'euronews',
    'euronews.com',
    'sky news',
    'news.sky.com',
    'the independent',
    'independent.co.uk',
    'the telegraph',
    'telegraph.co.uk',
    'cbc news',
    'cbc.ca',
    'ctv news',
    'ctvnews.ca',
    'global news',
    'globalnews.ca',
    'abc news australia',
    'abc.net.au',
    'sydney morning herald',
    'smh.com.au',
    'the age',
    'theage.com.au',
    'the australian financial review',
    'afr.com',
    'the hindu',
    'thehindu.com',
    'times of india',
    'indiatimes.com',
    'economic times',
    'economictimes.indiatimes.com',
    'hindustan times',
    'hindustantimes.com',
    'indian express',
    'indianexpress.com',
    'financial express',
    'financialexpress.com',
    'ndtv',
    'ndtv.com',
    'deccan herald',
    'deccanherald.com',
    'livemint',
    'livemint.com',
    'business standard',
    'business-standard.com',
    'moneycontrol',
    'moneycontrol.com',
    'the print',
    'theprint.in',
    'scroll',
    'scroll.in',
    'india today',
    'indiatoday.in',
    'outlook india',
    'outlookindia.com',
    'news18',
    'news18.com',
    'times now',
    'timesnownews.com',
    'wion',
    'wionews.com',
    'the week',
    'theweek.in',
    'the telegraph india',
    'telegraphindia.com',
    'the tribune india',
    'tribuneindia.com',
    'the statesman',
    'thestatesman.com',
    'ani',
    'aninews.in',
    'new indian express',
    'newindianexpress.com',
    'pib',
    'pib.gov.in',
    'south china morning post',
    'scmp.com',
    'the straits times',
    'straitstimes.com',
    'japan times',
    'japantimes.co.jp',
    'snopes',
    'snopes.com',
    'factcheck.org',
    'politifact',
    'politifact.com',
    'full fact',
    'fullfact.org',
)
ENV_PREFERRED_INDIA_NEWS_SOURCES = _parse_csv_list(os.getenv('PREFERRED_INDIA_NEWS_SOURCES', ''))
PREFERRED_INDIA_NEWS_SOURCES = _unique_items(
    DEFAULT_PREFERRED_INDIA_NEWS_SOURCES,
    ENV_PREFERRED_INDIA_NEWS_SOURCES,
)
ENV_TRUSTED_NEWS_SOURCES = _parse_csv_list(os.getenv('TRUSTED_NEWS_SOURCES', ''))
TRUSTED_NEWS_SOURCES = _unique_items(DEFAULT_TRUSTED_NEWS_SOURCES, ENV_TRUSTED_NEWS_SOURCES)
SOURCE_TRUST_WEIGHT = min(0.95, max(0.05, _parse_float(os.getenv('SOURCE_TRUST_WEIGHT', '0.35'), 0.35)))
MODEL_TRUST_WEIGHT = 1 - SOURCE_TRUST_WEIGHT
FLASK_DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'

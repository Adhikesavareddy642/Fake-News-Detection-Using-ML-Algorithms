from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fake_news_app.config import (
    GNEWS_API_KEY,
    GNEWS_API_URL,
    GNEWS_COUNTRY,
    GNEWS_INCLUDE_TRUSTED_IN_QUERY,
    GNEWS_QUERY_TRUSTED_TERMS_LIMIT,
    GNEWS_TIMEOUT_SECONDS,
    TRUSTED_NEWS_SOURCES,
)


class GNewsClientError(RuntimeError):
    """Base error for GNews integration issues."""


class GNewsConfigError(GNewsClientError):
    """Raised when GNews client configuration is missing/invalid."""


class GNewsRequestError(GNewsClientError):
    """Raised when GNews cannot be reached or returns malformed responses."""


def is_gnews_configured() -> bool:
    return bool(GNEWS_API_KEY)


def _sanitize_text(value, max_len: int = 2000) -> str:
    return str(value or '').strip()[:max_len]


def _extract_error_message(payload) -> str | None:
    if not isinstance(payload, dict):
        return None

    errors = payload.get('errors')
    if isinstance(errors, list) and errors:
        return str(errors[0])

    message = payload.get('message')
    if message:
        return str(message)
    return None


def _sanitize_query_token(value) -> str:
    token = str(value or '').strip().lower()
    token = token.replace('"', '')
    token = re.sub(r'\s+', ' ', token)
    return token


def _unique_tokens(values: list[str]) -> list[str]:
    ordered = {}
    for value in values:
        token = value.strip()
        if token:
            ordered[token] = None
    return list(ordered.keys())


def _build_source_terms(limit: int) -> list[str]:
    if limit <= 0:
        return []

    terms: list[str] = []
    for source in TRUSTED_NEWS_SOURCES:
        token = _sanitize_query_token(source)
        if not token:
            continue
        if '.' in token:
            terms.append(token)
            continue
        if ' ' in token:
            terms.append(f'"{token}"')
            continue
        if len(token) >= 4:
            terms.append(token)

    return _unique_tokens(terms)[:limit]


def _build_query_plan(base_query: str) -> list[str]:
    if not GNEWS_INCLUDE_TRUSTED_IN_QUERY:
        return [base_query]

    source_terms = _build_source_terms(GNEWS_QUERY_TRUSTED_TERMS_LIMIT)
    if not source_terms:
        return [base_query]

    source_clause = ' OR '.join(source_terms)
    enriched_query = f'({base_query}) AND ({source_clause})'
    if enriched_query == base_query:
        return [base_query]
    return [enriched_query, base_query]


def _fetch_payload(query: str, *, limit: int, language: str) -> dict:
    params = {
        'q': query,
        'lang': language,
        'max': limit,
        'sortby': 'publishedAt',
        'token': GNEWS_API_KEY,
    }
    if GNEWS_COUNTRY:
        params['country'] = GNEWS_COUNTRY
    request_url = f'{GNEWS_API_URL}?{urlencode(params)}'
    request_obj = Request(
        request_url,
        headers={
            'Accept': 'application/json',
            'User-Agent': 'TruthLens/1.0',
        },
    )

    try:
        with urlopen(request_obj, timeout=GNEWS_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode('utf-8', errors='replace')
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise GNewsConfigError(
                'GNews rejected the API key. Check GNEWS_API_KEY and try again.'
            ) from exc
        if exc.code == 429:
            raise GNewsRequestError('GNews rate limit reached. Please retry in a moment.') from exc
        raise GNewsRequestError(f'GNews request failed with status {exc.code}.') from exc
    except URLError as exc:
        raise GNewsRequestError(f'Failed to reach GNews: {exc.reason}') from exc

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise GNewsRequestError('GNews returned non-JSON data.') from exc

    error_message = _extract_error_message(payload)
    if error_message:
        raise GNewsRequestError(f'GNews error: {error_message}')

    if not isinstance(payload, dict):
        raise GNewsRequestError('GNews returned an invalid response payload.')
    return payload


def _extract_articles(payload: dict) -> tuple[list[dict], int]:
    raw_articles = payload.get('articles')
    if not isinstance(raw_articles, list):
        raise GNewsRequestError('GNews response did not include an articles list.')

    articles = []
    for entry in raw_articles:
        if not isinstance(entry, dict):
            continue

        title = _sanitize_text(entry.get('title'), max_len=400)
        description = _sanitize_text(entry.get('description'), max_len=800)
        content = _sanitize_text(entry.get('content'), max_len=1400)
        published_at = _sanitize_text(entry.get('publishedAt'), max_len=80)
        url = _sanitize_text(entry.get('url'), max_len=2000)
        source = entry.get('source') if isinstance(entry.get('source'), dict) else {}
        source_name = _sanitize_text(source.get('name'), max_len=140)

        analysis_text = ' '.join(part for part in (title, description, content) if part).strip()
        if not analysis_text:
            continue

        articles.append(
            {
                'title': title,
                'description': description,
                'content': content,
                'published_at': published_at,
                'url': url,
                'source_name': source_name,
                'analysis_text': analysis_text,
            }
        )

    total_articles = payload.get('totalArticles')
    if not isinstance(total_articles, int):
        total_articles = len(articles)
    return articles, total_articles


def search_news(query: str, *, limit: int = 5, language: str = 'en') -> dict:
    if not is_gnews_configured():
        raise GNewsConfigError(
            'GNews API key is missing. Set GNEWS_API_KEY before using real-time search.'
        )

    text_query = str(query or '').strip()
    if not text_query:
        raise GNewsClientError('Please enter a search query.')

    safe_limit = max(1, min(int(limit), 10))
    queries = _build_query_plan(text_query)

    combined_articles = []
    seen_ids = set()
    total_articles = 0
    last_error = None

    for request_query in queries:
        remaining = safe_limit - len(combined_articles)
        if remaining <= 0:
            break

        try:
            payload = _fetch_payload(request_query, limit=remaining, language=language)
            fetched_articles, fetched_total = _extract_articles(payload)
        except GNewsConfigError:
            raise
        except GNewsRequestError as exc:
            last_error = exc
            continue

        total_articles = max(total_articles, fetched_total)
        for article in fetched_articles:
            article_id = article['url'] or (
                f"{article['title']}|{article['published_at']}|{article['source_name']}"
            )
            if article_id in seen_ids:
                continue
            seen_ids.add(article_id)
            combined_articles.append(article)
            if len(combined_articles) >= safe_limit:
                break

    if not combined_articles and last_error is not None:
        raise last_error

    return {'query': text_query, 'total_articles': total_articles, 'articles': combined_articles}

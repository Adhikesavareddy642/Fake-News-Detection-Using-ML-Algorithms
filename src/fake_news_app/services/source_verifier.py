from __future__ import annotations

import re
from urllib.parse import urlparse

from fake_news_app.config import MODEL_TRUST_WEIGHT, SOURCE_TRUST_WEIGHT, TRUSTED_NEWS_SOURCES

TRUSTED_SOURCE_SCORE = 1.0
UNTRUSTED_SOURCE_SCORE = 0.45


def _normalize_text(value) -> str:
    text = str(value or '').strip().lower()
    return re.sub(r'\s+', ' ', text)


def _extract_domain(url: str) -> str:
    try:
        domain = urlparse(str(url or '')).netloc.lower()
    except Exception:
        return ''
    return domain[4:] if domain.startswith('www.') else domain


def _is_trusted_source(source_name: str, source_domain: str) -> bool:
    normalized_name = _normalize_text(source_name)
    domain = _normalize_text(source_domain)
    for trusted in TRUSTED_NEWS_SOURCES:
        rule = _normalize_text(trusted)
        if not rule:
            continue
        if '.' in rule:
            if domain == rule or domain.endswith(f'.{rule}'):
                return True
        if normalized_name == rule or rule in normalized_name:
            return True
    return False


def _derive_model_real_probability(model_label: str, model_confidence) -> float:
    if model_confidence is None:
        return 0.5

    confidence = float(model_confidence)
    if str(model_label or '').strip().lower().startswith('real'):
        return confidence
    return 1 - confidence


def derive_source_verification(model_label: str, model_confidence, source_name: str, article_url: str) -> dict:
    source_domain = _extract_domain(article_url)
    trusted_source = _is_trusted_source(source_name, source_domain)

    source_score = TRUSTED_SOURCE_SCORE if trusted_source else UNTRUSTED_SOURCE_SCORE
    model_real_probability = _derive_model_real_probability(model_label, model_confidence)
    verification_score = (model_real_probability * MODEL_TRUST_WEIGHT) + (source_score * SOURCE_TRUST_WEIGHT)

    if trusted_source:
        verification_status = 'Verified: Trusted source'
        final_label = 'Real News'
        is_verified = True
    elif verification_score >= 0.65:
        verification_status = 'Likely Real'
        final_label = 'Real News'
        is_verified = False
    elif verification_score <= 0.35:
        verification_status = 'Likely Fake'
        final_label = 'Fake News'
        is_verified = False
    else:
        verification_status = 'Needs Verification'
        final_label = model_label
        is_verified = False

    return {
        'trusted_source': trusted_source,
        'source_domain': source_domain,
        'source_score': round(source_score, 4),
        'model_real_probability': round(model_real_probability, 4),
        'verification_score': round(verification_score, 4),
        'verification_status': verification_status,
        'is_verified': is_verified,
        'final_label': final_label,
    }

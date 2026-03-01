from __future__ import annotations

import json

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from fake_news_app.config import MODEL_PATH
from fake_news_app.services.predictor import get_model, predict_from_text, refresh_model
from fake_news_app.services.trainer import (
    load_default_dataset,
    parse_rows,
    save_custom_samples,
    train_and_export_model,
)

bp = Blueprint('main', __name__)


@bp.get('/')
def home():
    return render_template(
        'index.html',
        model_found=get_model() is not None,
        model_path=str(MODEL_PATH),
    )


@bp.get('/predict')
def predict_ui():
    return redirect(url_for('main.home'))


@bp.get('/train')
def train_ui():
    return render_template(
        'train.html',
        model_found=get_model() is not None,
        model_path=str(MODEL_PATH),
    )


@bp.get('/train-ui')
def train_ui_legacy():
    return redirect(url_for('main.train_ui'))


@bp.post('/api/predict')
@bp.post('/predict')
def predict_api():
    payload = request.get_json(silent=True) or {}
    text = (payload.get('text') or '').strip()

    if not text:
        return jsonify({'ok': False, 'error': 'Please enter article text.'}), 400

    try:
        result = predict_from_text(text)
    except FileNotFoundError:
        return jsonify(
            {
                'ok': False,
                'error': f'Model file not found at {MODEL_PATH}. Train and export your model first.',
            }
        ), 400
    except RuntimeError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'Prediction failed: {exc}'}), 500

    return jsonify({'ok': True, **result})


@bp.post('/api/train')
@bp.post('/train')
def train_api():
    samples_json = (request.form.get('samples_json') or '').strip()

    try:
        base_texts, base_labels = load_default_dataset()
        if not samples_json:
            raise RuntimeError('Please add custom samples in the form before training.')

        rows = json.loads(samples_json)
        if not isinstance(rows, list):
            raise RuntimeError('Invalid form samples payload.')

        custom_texts, custom_labels = parse_rows(rows)
        source = 'form samples + existing dataset'

        texts = list(base_texts)
        labels = list(base_labels)
        texts.extend(custom_texts)
        labels.extend(custom_labels)

        report = train_and_export_model(texts, labels)
        save_custom_samples(custom_texts, custom_labels)
        refresh_model()
    except (RuntimeError, json.JSONDecodeError) as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'Training failed: {exc}'}), 500

    return jsonify(
        {
            'ok': True,
            'message': (
                f"Training complete using {source}. "
                f"Validation accuracy: {report['accuracy']:.4f}. "
                f"Samples used: {report['samples']} "
                f"(base: {len(base_texts)}, added: {len(custom_texts)})."
            ),
            'base_samples': len(base_texts),
            'added_samples': len(custom_texts),
            **report,
        }
    )

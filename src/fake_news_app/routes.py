from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from fake_news_app.config import MODEL_PATH
from fake_news_app.services.predictor import get_model, predict_from_text

bp = Blueprint('main', __name__)


@bp.get('/')
def index():
    return render_template(
        'index.html',
        model_found=get_model() is not None,
        model_path=str(MODEL_PATH),
    )


@bp.post('/predict')
def predict():
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

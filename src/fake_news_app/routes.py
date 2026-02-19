from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, render_template, request

from fake_news_app.config import FORM_DATASET_PATH, MODEL_PATH
from fake_news_app.services.predictor import get_model, predict_from_text, reload_model
from fake_news_app.services.trainer import (
    append_samples_to_csv,
    load_training_data_from_csv,
    train_and_export_from_lists,
)

bp = Blueprint('main', __name__)


@bp.get('/')
def index():
    return render_template(
        'index.html',
        model_found=get_model() is not None,
        model_path=str(MODEL_PATH),
    )


@bp.route("/train-ui", methods=["GET", "POST"])
def train_ui():
    if request.method == "GET":
        return render_template("train.html")

    try:
        test_size = float(request.form.get("test_size") or "0.2")
    except ValueError:
        return render_template("train.html", error="test_size must be a decimal, e.g. 0.2")

    try:
        random_state = int(request.form.get("random_state") or "42")
    except ValueError:
        return render_template("train.html", error="random_state must be an integer.")

    fake_samples_raw = request.form.get("fake_samples") or ""
    real_samples_raw = request.form.get("real_samples") or ""
    fake_samples = [line.strip() for line in fake_samples_raw.splitlines() if line.strip()]
    real_samples = [line.strip() for line in real_samples_raw.splitlines() if line.strip()]

    if not fake_samples and not real_samples:
        return render_template("train.html", error="Please provide at least one sample in fake or real.")

    try:
        new_texts = fake_samples + real_samples
        new_labels = [0] * len(fake_samples) + [1] * len(real_samples)
        appended_rows = append_samples_to_csv(
            csv_path=FORM_DATASET_PATH,
            texts=new_texts,
            labels=new_labels,
            text_column="text",
            label_column="label",
        )
        texts, labels = load_training_data_from_csv(
            csv_path=FORM_DATASET_PATH,
            text_column="text",
            label_column="label",
        )
        metrics = train_and_export_from_lists(
            texts=texts,
            labels=labels,
            test_size=test_size,
            random_state=random_state,
        )
        reload_model()
    except Exception as exc:
        return render_template("train.html", error=f"Training failed: {exc}")

    result = {
        "message": "Model training completed (new samples appended to existing training data).",
        "dataset_path": str(FORM_DATASET_PATH),
        "appended_rows": appended_rows,
        "submitted_fake_rows": len(fake_samples),
        "submitted_real_rows": len(real_samples),
        "trained_on_rows": len(texts),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }
    return render_template("train.html", result=result)


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

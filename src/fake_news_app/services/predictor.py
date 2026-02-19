from __future__ import annotations

import joblib

from fake_news_app.config import MODEL_PATH

LABEL_MAP = {0: 'Fake News', 1: 'Real News'}
STRING_LABEL_MAP = {
    '0': 'Fake News',
    '1': 'Real News',
    'FAKE': 'Fake News',
    'REAL': 'Real News',
}

model = None
model_load_error = None


def load_model():
    """Load a serialized sklearn-compatible pipeline."""
    if not MODEL_PATH.exists():
        return None, None
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f'Failed to load model from {MODEL_PATH}: {exc}'


def get_model():
    """Get model, retrying load if file appears after startup."""
    global model, model_load_error
    if model is None and MODEL_PATH.exists():
        model, model_load_error = load_model()
    return model


def normalize_label(raw_pred):
    """Normalize different model label formats into UI labels."""
    if isinstance(raw_pred, int):
        return LABEL_MAP.get(raw_pred, str(raw_pred))
    if isinstance(raw_pred, float) and raw_pred.is_integer():
        return LABEL_MAP.get(int(raw_pred), str(raw_pred))
    key = str(raw_pred).strip().upper()
    return STRING_LABEL_MAP.get(key, str(raw_pred))


def run_inference(model_obj, text: str):
    """Run model prediction and optional confidence extraction."""
    pred_raw = model_obj.predict([text])[0]
    label = normalize_label(pred_raw)
    confidence = None

    if hasattr(model_obj, 'predict_proba'):
        probs = model_obj.predict_proba([text])[0]
        confidence = float(max(probs))

    return {'label': label, 'confidence': confidence}


def predict_from_text(text: str):
    """Wrapper method that calls inference."""
    model_obj = get_model()
    if model_obj is None:
        if model_load_error:
            raise RuntimeError(model_load_error)
        raise FileNotFoundError(f'Model file not found at {MODEL_PATH}.')
    return run_inference(model_obj, text)


model, model_load_error = load_model()

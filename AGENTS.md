# Repository Guidelines

## Project Structure & Module Organization
- Core application code lives in `src/fake_news_app/`.
- HTTP routes are in `src/fake_news_app/routes.py`; model logic is in `src/fake_news_app/services/` (`predictor.py`, `trainer.py`).
- Entry point: `app.py` (adds `src/` to `PYTHONPATH` and starts Flask).
- Training script: `scripts/train_and_export_model.py`.
- Data and artifacts:
  - `datasets/fake_or_real_news.csv` (main training dataset)
  - `artifacts/fake_news_pipeline.joblib` and `artifacts/custom_samples.json` (generated outputs)
- UI assets: `templates/` and `static/`.
- `Code/` and `Data Set/` contain legacy notebook/data copies; prefer `src/`, `scripts/`, and `datasets/` for active work.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (create/activate virtualenv).
- `pip install -r requirements.txt` (install Flask + ML dependencies).
- `python app.py` (run local web app at `http://127.0.0.1:5000`).
- `python scripts/train_and_export_model.py` (train model and write artifact to `artifacts/`).
- `FLASK_DEBUG=1 python app.py` (run with debug reload enabled).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear, typed function signatures where practical.
- Use `snake_case` for files, functions, and variables; keep route handlers short and delegate logic to `services/`.
- Prefer small, single-purpose helpers (see `parse_rows`, `_normalize_label`).
- Keep user-facing errors actionable and explicit.

## Testing Guidelines
- No automated test suite is currently configured.
- For new features, add `pytest` tests under `tests/` (e.g., `tests/test_trainer.py`, `tests/test_routes.py`).
- Minimum expectation for PRs: cover one success path and one failure path for each changed service/route.
- Run tests with `pytest -q` once tests are added.

## Commit & Pull Request Guidelines
- Existing history uses short, imperative commit messages (e.g., `Restructure project to src layout...`).
- Keep commits focused: one concern per commit (routing, training logic, UI, etc.).
- PRs should include:
  - concise summary of behavior changes,
  - linked issue/task (if available),
  - screenshots or curl examples for UI/API changes,
  - notes on dataset/artifact impacts.

## Security & Configuration Tips
- Do not commit large generated artifacts or sensitive datasets beyond tracked project files.
- Validate CSV inputs and labels (`FAKE`/`REAL` or `0`/`1`) before training.

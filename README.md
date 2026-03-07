# Fake News Detection

This project includes a Flask web app for fake-news prediction and a training/export script for building the model artifact.

## Package Structure
```text
.
├── app.py                          # App entrypoint (wrapper)
├── src/
│   └── fake_news_app/              # Application package
│       ├── __init__.py             # create_app()
│       ├── config.py               # paths + runtime config
│       ├── routes.py               # HTTP routes
│       └── services/
│           ├── __init__.py
│           └── predictor.py        # model load + inference logic
├── scripts/
│   └── train_and_export_model.py   # model training/export
├── datasets/
│   └── fake_or_real_news.csv
├── notebooks/
│   └── Final_Project (1).ipynb
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── artifacts/
│   └── fake_news_pipeline.joblib
└── requirements.txt
```

## Run the UI
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create your env file:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` and set your real keys (for example `GNEWS_API_KEY`).
5. Start the app:
   ```bash
   python app.py
   ```
6. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Real-Time News Search (GNews)
- Put `GNEWS_API_KEY` in `.env` (see `.env.example`).
- Default country is India via `GNEWS_COUNTRY=in` (override in `.env` if needed).
- Open `http://127.0.0.1:5000/live-news` to search live topics and classify results.
- API endpoint:
  - `GET /api/live-detect?query=ai&limit=5`
  - `POST /api/live-detect` with JSON body: `{"query":"ai","limit":5}`
- Source-based verification:
  - The app ships with a broad built-in trusted-source list (major global outlets + fact-checking sites).
  - Indian domains are also included by default (for example: `thehindu.com`, `indiatimes.com`, `indianexpress.com`, `ndtv.com`, `livemint.com`).
  - Add extra sources in `.env` using `TRUSTED_NEWS_SOURCES` (comma-separated); these are merged with defaults.
  - `SOURCE_TRUST_WEIGHT` controls how much source trust contributes to the final verification score.
  - For trusted sources, live detection marks the article as verified and sets final label to `Real News`.

## Train/Export Model
Run:
```bash
python scripts/train_and_export_model.py
```

The script reads dataset from:
`datasets/fake_or_real_news.csv`

The model artifact is written to:
`artifacts/fake_news_pipeline.joblib`

## Train from UI (Your Own Data)
The web UI now supports training with your own form data:
- Add samples in the **Train Your Own Data** form (`text` + `label`)
- Click **Train Custom Data**
- New samples are appended to `datasets/fake_or_real_news.csv` content in-memory for training
- Custom form samples are also saved for exact-match priority during prediction

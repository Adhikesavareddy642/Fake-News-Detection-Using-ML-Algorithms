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
3. Start the app:
   ```bash
   python app.py
   ```
4. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Train/Export Model
Run:
```bash
python scripts/train_and_export_model.py
```

The script reads dataset from:
`datasets/fake_or_real_news.csv`

The model artifact is written to:
`artifacts/fake_news_pipeline.joblib`

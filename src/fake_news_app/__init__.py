from __future__ import annotations

from flask import Flask

from fake_news_app.config import BASE_DIR
from fake_news_app.routes import bp


def create_app():
    templates_dir = BASE_DIR / 'templates'
    static_dir = BASE_DIR / 'static'

    app = Flask(
        __name__,
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
    )
    app.register_blueprint(bp)
    return app

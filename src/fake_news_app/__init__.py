from __future__ import annotations

from flask import Flask

from fake_news_app.routes import bp


def create_app():
    app = Flask(
        __name__,
        template_folder='../../templates',
        static_folder='../../static',
    )
    app.register_blueprint(bp)
    return app

import os
import json
import logging
from datetime import datetime

import redis
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

load_dotenv()


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def create_app() -> Flask:
    app = Flask(__name__, static_folder="../frontend", static_url_path="")

    app.secret_key = os.environ.get("SECRET_KEY", "exam_portal_secret_2026")

    _raw_origins = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5000,http://127.0.0.1:5000",
    )
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)

    IS_PRODUCTION = os.environ.get("FLASK_ENV", "development") == "production"
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=IS_PRODUCTION,
    )

    # ── Redis client ──
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    app.config["REDIS_CLIENT"] = redis.from_url(redis_url, decode_responses=True)

    # ── JSON logging ──
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("App created", extra={"env": os.environ.get("FLASK_ENV", "development")})

    # ── Initialise database tables ──
    from backend.database import init_db
    init_db()

    # ── Register blueprints ──
    from backend.auth.routes import auth_bp
    from backend.exam.routes import exam_bp
    from backend.proctor.routes import proctor_bp
    from backend.teacher.routes import teacher_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(exam_bp)
    app.register_blueprint(proctor_bp)
    app.register_blueprint(teacher_bp)

    return app

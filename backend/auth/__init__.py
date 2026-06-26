from flask import Blueprint

auth_bp = Blueprint("auth", __name__)

from backend.auth.routes import *  # noqa: F401, E402

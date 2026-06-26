from flask import Blueprint

exam_bp = Blueprint("exam", __name__)

from backend.exam.routes import *  # noqa: F401, E402

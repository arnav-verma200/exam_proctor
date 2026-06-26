from flask import Blueprint

teacher_bp = Blueprint("teacher", __name__)

from backend.teacher.routes import *  # noqa: F401, E402

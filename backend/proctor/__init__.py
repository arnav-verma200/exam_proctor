from flask import Blueprint

proctor_bp = Blueprint("proctor", __name__)

from backend.proctor.routes import *  # noqa: F401, E402

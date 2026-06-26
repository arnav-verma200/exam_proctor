from flask import redirect, request, session, send_from_directory

from backend.auth import auth_bp
from backend.models.schemas import LoginRequest, TeacherLoginRequest
from backend.services.auth_service import authenticate_candidate, authenticate_teacher
from backend.services.exam_service import create_session, get_session, time_left
from backend.services.event_service import log_event


def _ok(data: dict | None = None, msg: str = "OK"):
    body = {"status": "success", "message": msg}
    if data:
        body.update(data)
    from flask import jsonify
    return jsonify(body), 200


def _err(msg: str, code: int = 400):
    from flask import jsonify
    return jsonify({"status": "error", "message": msg}), code


@auth_bp.route("/api/login", methods=["POST"])
def candidate_login():
    d = LoginRequest(**request.get_json(force=True) or {})
    app_no = str(d.app_number).strip()
    pw = str(d.password).strip()

    cand = authenticate_candidate(app_no, pw)
    if not cand:
        return _err("Invalid application number or password.", 401)

    sess = get_session(app_no)
    if sess.get("submitted"):
        return _err("Your exam has already been submitted.", 403)

    session["app_number"] = app_no
    session["role"] = "candidate"

    create_session(app_no, cand["name"])
    log_event(app_no, "Candidate Login", 100)

    from backend.services.exam_service import EXAM_CONFIG
    return _ok(
        {"candidate": cand, "exam": EXAM_CONFIG},
        "Login successful",
    )


@auth_bp.route("/api/logout", methods=["GET", "POST"])
def candidate_logout():
    session.clear()
    return _ok(msg="Logged out")


@auth_bp.route("/api/session", methods=["GET"])
def check_session():
    if "app_number" not in session:
        return _err("No active session.", 401)

    app_no = session["app_number"]
    cand = authenticate_candidate(app_no, "")
    sess = get_session(app_no)

    return _ok({
        "app_number": app_no,
        "name": cand.get("name", "Unknown") if cand else "Unknown",
        "exam_started": sess.get("started", False),
        "exam_submitted": sess.get("submitted", False),
        "time_remaining": time_left(app_no),
    })


@auth_bp.route("/api/teacher/login", methods=["POST"])
def teacher_login():
    d = TeacherLoginRequest(**request.get_json(force=True) or {})
    user = authenticate_teacher(d.username, d.password)
    if not user:
        return _err("Invalid credentials.", 401)

    session["teacher"] = user
    session["role"] = "teacher"
    return _ok({"username": user}, "Teacher login successful")


@auth_bp.route("/api/teacher/logout", methods=["POST"])
def teacher_logout():
    session.clear()
    return _ok(msg="Logged out")


@auth_bp.route("/")
def index():
    return redirect("/connected_portal.html")

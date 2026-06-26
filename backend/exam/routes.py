from flask import request, session, jsonify

from backend.exam import exam_bp
from backend.models.schemas import ExamStartRequest, SaveAnswerRequest, SubmitRequest
from backend.services.exam_service import (
    EXAM_CONFIG,
    get_session,
    start_exam,
    save_answer,
    submit_exam,
    get_exam_status,
    time_left,
)
from backend.services.event_service import log_event


def _ok(data: dict | None = None, msg: str = "OK"):
    body = {"status": "success", "message": msg}
    if data:
        body.update(data)
    return jsonify(body), 200


def _err(msg: str, code: int = 400):
    return jsonify({"status": "error", "message": msg}), code


@exam_bp.route("/api/exam/info", methods=["GET"])
def exam_info():
    return _ok({"exam": EXAM_CONFIG})


@exam_bp.route("/api/exam/start", methods=["POST"])
def exam_start():
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    d = ExamStartRequest(**request.get_json(force=True) or {})

    if not d.agreed:
        return _err("You must agree to the instructions.")

    sess = get_session(app_no)
    if sess.get("started"):
        return _ok(
            {"already_started": True, "time_remaining": time_left(app_no)},
            "Exam already in progress",
        )

    result = start_exam(app_no, EXAM_CONFIG["duration_mins"])
    log_event(app_no, "Exam Started", 100)

    return _ok(result, "Exam started")


@exam_bp.route("/api/exam/answer", methods=["POST"])
def save_answer_route():
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    sess = get_session(app_no)

    if not sess.get("started"):
        return _err("Exam not started.")
    if sess.get("submitted"):
        return _err("Exam already submitted.")
    if time_left(app_no) <= 0:
        return _err("Time is up.")

    d = SaveAnswerRequest(**request.get_json(force=True) or {})
    if not d.question_id:
        return _err("question_id is required.")

    save_answer(app_no, d.question_id, d.answer)
    return _ok({"saved": d.question_id})


@exam_bp.route("/api/exam/submit", methods=["POST"])
def exam_submit():
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    d = SubmitRequest(**request.get_json(force=True) or {})

    total = submit_exam(app_no, d.answers)
    if total is None:
        sess = get_session(app_no)
        if not sess.get("started"):
            return _err("Exam not started.")
        return _err("Exam already submitted.")

    log_event(app_no, "Exam Submitted", 100)
    return _ok({"total_answered": total}, "Exam submitted successfully")


@exam_bp.route("/api/exam/status", methods=["GET"])
def exam_status():
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    return _ok(get_exam_status(app_no))

from flask import request, session, jsonify

from backend.teacher import teacher_bp
from backend.services.event_service import events_for, EVENT_META
from backend.services.exam_service import CANDIDATE_SESSIONS, time_left


def _ok(data: dict | None = None, msg: str = "OK"):
    body = {"status": "success", "message": msg}
    if data:
        body.update(data)
    return jsonify(body), 200


def _err(msg: str, code: int = 400):
    return jsonify({"status": "error", "message": msg}), code


@teacher_bp.route("/api/teacher/students", methods=["GET"])
def teacher_students():
    students = []
    for app_no, sess in CANDIDATE_SESSIONS.items():
        evts = events_for(app_no)
        risk = sum(e["impact"] for e in evts)
        last_event = evts[0]["event_type"] if evts else "Logged In"

        if sess.get("submitted"):
            status_label = "Submitted"
        elif sess.get("started"):
            status_label = last_event
        else:
            status_label = "Waiting"

        students.append({
            "app_number": app_no,
            "name": sess["name"],
            "started": sess.get("started", False),
            "submitted": sess.get("submitted", False),
            "risk_score": min(100, risk),
            "event_count": len(evts),
            "last_event": status_label,
            "time_remaining": time_left(app_no),
            "answered_count": len(sess.get("answers", {})),
        })

    students.sort(key=lambda s: s["risk_score"], reverse=True)
    return _ok({"students": students, "total": len(students)})

import base64
import time

from flask import Response, jsonify, request, current_app

from backend.proctor import proctor_bp
from backend.models.schemas import ProctorEventRequest, SnapshotRequest, SetCandidateRequest
from backend.services.event_service import (
    log_event,
    filter_events,
    export_csv,
    add_snapshot,
    filter_snapshots,
    PROCTOR_EVENTS,
    SNAPSHOTS,
)
from backend.services.proctor_service import build_summary, calculate_risk_from_events
from backend.services.vision_service import (
    store_frame,
    set_active_candidate,
    clear_active_candidate,
    ensure_detection_thread,
    get_latest_annotated_frame,
    vision_state,
    vision_lock,
    _make_placeholder,
)


def _ok(data: dict | None = None, msg: str = "OK"):
    body = {"status": "success", "message": msg}
    if data:
        body.update(data)
    return jsonify(body), 200


def _err(msg: str, code: int = 400):
    return jsonify({"status": "error", "message": msg}), code


def _rate_limit(app_number: str) -> bool:
    r = current_app.config.get("REDIS_CLIENT")
    if not r:
        return False
    key = f"ratelimit:frame:{app_number}"
    now = int(time.time())
    window = 1
    max_reqs = 10
    pipeline = r.pipeline()
    pipeline.zremrangebyscore(key, "-inf", now - window)
    pipeline.zcard(key)
    pipeline.zadd(key, {str(now): now})
    pipeline.expire(key, window + 1)
    _, count, _, _ = pipeline.execute()
    return int(count) >= max_reqs


# Start the detection thread on module import (lazy via first request)
ensure_detection_thread()


@proctor_bp.before_request
def _before_proctor_request():
    ensure_detection_thread()


@proctor_bp.route("/proctor/frame/<app_number>", methods=["POST"])
def receive_frame(app_number):
    if _rate_limit(app_number):
        return jsonify({"ok": False, "error": "rate limit exceeded"}), 429

    try:
        ct = request.content_type or ""
        if "json" in ct:
            data = request.get_json(force=True) or {}
            img_b64: str = data.get("image", "")
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            raw = base64.b64decode(img_b64)
        else:
            raw = request.get_data()

        if not raw:
            return jsonify({"ok": False, "error": "empty body"}), 400

        store_frame(app_number, raw)

        return jsonify({"ok": True}), 200

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _generate_mjpeg():
    while True:
        frame = get_latest_annotated_frame()
        if frame is None:
            frame = _make_placeholder()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@proctor_bp.route("/proctor/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@proctor_bp.route("/proctor/set_candidate", methods=["POST"])
def proctor_set_candidate():
    d = SetCandidateRequest(**request.get_json(force=True) or {})
    set_active_candidate(d.app_number, d.student_name)
    return jsonify({"status": "ok"})


@proctor_bp.route("/proctor/clear_candidate", methods=["POST"])
def proctor_clear_candidate():
    clear_active_candidate()
    return jsonify({"status": "ok"})


@proctor_bp.route("/proctor/status")
def proctor_status():
    with vision_lock:
        return jsonify({
            "app_number": vision_state["app_number"],
            "student_name": vision_state["student_name"],
            "face_count": vision_state["face_count"],
            "phone_detected": vision_state["phone_detected"],
            "status_text": vision_state["status_text"],
        })


@proctor_bp.route("/proctor/ping")
def proctor_ping():
    return jsonify({"ok": True})


@proctor_bp.route("/api/proctor/event", methods=["POST"])
def proctor_log():
    d = ProctorEventRequest(**request.get_json(force=True) or {})
    log_event(d.app_number, d.event_type, d.confidence, d.impact, note=d.note)
    return _ok(msg="Event logged")


@proctor_bp.route("/api/proctor/events", methods=["GET"])
def proctor_events():
    student_filter = request.args.get("student", "").lower()
    type_filter = request.args.get("type", "")
    filtered = filter_events(student_filter, type_filter)
    return _ok({"events": filtered, "total": len(filtered)})


@proctor_bp.route("/api/proctor/summary", methods=["GET"])
def proctor_summary():
    return _ok(build_summary())


@proctor_bp.route("/api/proctor/export", methods=["GET"])
def proctor_export():
    return export_csv()


@proctor_bp.route("/api/proctor/snapshot", methods=["POST"])
def proctor_snapshot():
    d = SnapshotRequest(**request.get_json(force=True) or {})
    if not d.image or not d.image.startswith("data:image"):
        return _err("Invalid image data.")
    add_snapshot(d.app_number, d.student_name, d.event_type, d.image)
    return _ok(msg="Snapshot saved")


@proctor_bp.route("/api/proctor/snapshots", methods=["GET"])
def proctor_snapshots():
    student_filter = request.args.get("student", "").lower()
    type_filter = request.args.get("type", "")
    filtered = filter_snapshots(student_filter, type_filter)
    return _ok({"snapshots": filtered, "total": len(filtered)})

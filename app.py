"""
=============================================================
  app.py  —  Exam Portal + Proctor Vision Engine (MERGED)

  Single-process deployment. Run with ONE worker only because
  detection state lives in process memory (not Redis/DB).

  Frame pipeline:
    Browser  →  POST /proctor/frame/<app_number>  (JPEG bytes)
    Server   →  OpenCV + MediaPipe face detection
    Teacher  ←  GET  /proctor/video_feed          (MJPEG stream)
    Flags    →  log_event() called directly in-process

  Local:   python app.py
  Deploy:  gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
           ^^^ workers=1 is REQUIRED — shared state is not safe across workers
=============================================================
"""

import base64
import csv
import hashlib
import io
import os
import threading
import time
import uuid
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, session
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "exam_portal_secret_2026")

# Parse allowed CORS origins from env var (comma-separated)
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
    # Secure flag only in production (requires HTTPS)
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
)

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — VISION ENGINE
#  Receives JPEG frames from the candidate's browser,
#  runs MediaPipe face detection + heuristic phone detection,
#  and produces an annotated MJPEG stream for the teacher.
# ═══════════════════════════════════════════════════════════════

# ── MediaPipe face detector (lazy init to avoid crashes on import) ──

_face_detector: mp.solutions.face_detection.FaceDetection | None = None
_face_detector_lock = threading.Lock()


def _get_face_detector() -> mp.solutions.face_detection.FaceDetection:
    """Return a shared MediaPipe FaceDetection instance, initialised once."""
    global _face_detector
    with _face_detector_lock:
        if _face_detector is None:
            _face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,          # full-range model (up to ~5 m)
                min_detection_confidence=0.5,
            )
    return _face_detector


# Seconds to wait before re-firing the same event type
VISION_COOLDOWN = {
    "Face Not Detected": 8,
    "Multiple Faces": 8,
    "Phone Detected": 10,
}

# If no frame is received within this many seconds, show a placeholder
FRAME_TIMEOUT = 10

# ── Per-candidate raw frame storage ──
# { app_number: { "raw_jpeg": bytes, "last_seen": float } }
candidate_frames: dict[str, dict] = {}
frames_lock = threading.Lock()

# ── Shared vision state (read by detection loop, written atomically) ──
vision_state: dict = {
    "app_number": None,       # currently monitored candidate
    "student_name": "Unknown",
    "face_count": 0,
    "phone_detected": False,
    "status_text": "Waiting for candidate…",
    "last_flag": {},          # event_type → timestamp of last fire
    "output_frame": None,     # annotated JPEG bytes served via MJPEG
}
vision_lock = threading.Lock()

# ── Detection thread bookkeeping ──
_detection_thread_started = False
_detection_thread_lock = threading.Lock()


# ─── Helpers ─────────────────────────────────────────────────

def _make_placeholder(text: str = "Waiting for candidate…") -> bytes:
    """Return a dark JPEG frame with centred status text."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (130, 130, 130), 2)
    cv2.putText(
        img,
        datetime.now().strftime("%H:%M:%S"),
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (70, 70, 70),
        1,
    )
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes()


def _detect_phone(frame_bgr: np.ndarray, face_bboxes: list[tuple]) -> tuple[bool, int]:
    """
    Heuristic phone detector using edge contours + aspect ratio.

    A phone-sized rectangle (3 %–35 % of frame area, aspect 1.5–2.5)
    that does NOT overlap any detected face is flagged.

    Returns (detected: bool, confidence: int 0-100).
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    dilated = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (0.03 * w * h < area < 0.35 * w * h):
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 5 or rh < 5:
            continue
        # Phone aspect ratio is roughly 1.5–2.5
        if not (1.5 <= max(rw, rh) / min(rw, rh) <= 2.5):
            continue
        cx, cy = int(rect[0][0]), int(rect[0][1])
        # Discard rectangles that overlap an already-detected face
        overlaps_face = any(
            fx <= cx <= fx + fw and fy <= cy <= fy + fh
            for fx, fy, fw, fh in face_bboxes
        )
        if not overlaps_face:
            confidence = min(95, int(60 + (area / (w * h)) * 200))
            return True, confidence

    return False, 0


def _notify_proctor_event(event_type: str, confidence: int) -> None:
    """
    Log a vision violation directly into the in-process event store.

    Respects per-event-type cooldowns so the log is not spammed.
    Must be called from a daemon thread (it accesses shared state).
    """
    app_no = vision_state["app_number"]
    if not app_no:
        return

    now = time.time()
    last_fired = vision_state["last_flag"].get(event_type, 0)
    cooldown = VISION_COOLDOWN.get(event_type, 5)
    if now - last_fired < cooldown:
        return  # still within cooldown window

    vision_state["last_flag"][event_type] = now
    log_event(app_no, event_type, confidence)
    print(f"[vision] ⚠  {event_type} ({confidence}%) → {app_no}")


def _detection_loop() -> None:
    """
    Background thread: processes the latest browser frame from the active
    candidate, runs face + phone detection, and writes an annotated JPEG
    to vision_state["output_frame"] for the MJPEG stream.
    """
    while True:
        app_no = vision_state["app_number"]
        raw_jpeg: bytes | None = None
        last_seen: float = 0.0

        # Grab the most recent frame for the active candidate
        if app_no:
            with frames_lock:
                slot = candidate_frames.get(app_no)
                if slot:
                    raw_jpeg = slot["raw_jpeg"]
                    last_seen = slot["last_seen"]

        # Show placeholder if no frame or frame is stale
        if raw_jpeg is None or (time.time() - last_seen) > FRAME_TIMEOUT:
            placeholder_text = (
                f"Waiting for {vision_state['student_name']}…"
                if app_no
                else "No candidate active"
            )
            with vision_lock:
                vision_state["output_frame"] = _make_placeholder(placeholder_text)
            time.sleep(0.1)
            continue

        # Decode JPEG → BGR numpy array
        try:
            arr = np.frombuffer(raw_jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as exc:
            print(f"[vision] Frame decode error: {exc}")
            time.sleep(0.05)
            continue

        # Mirror the frame (selfie-style)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # ── Face detection ──
        results = _get_face_detector().process(rgb)
        face_count = 0
        face_bboxes: list[tuple] = []

        if results.detections:
            face_count = len(results.detections)
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = int(bb.width * w)
                bh = int(bb.height * h)
                face_bboxes.append((x, y, bw, bh))
                # Green box for single face, blue for multiple
                box_color = (0, 220, 80) if face_count == 1 else (0, 80, 220)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
                cv2.putText(
                    frame,
                    f"Face {int(det.score[0] * 100)}%",
                    (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                )

        # ── Phone detection ──
        phone_detected, phone_conf = _detect_phone(frame, face_bboxes)

        # ── Update status and fire events ──
        if face_count == 0:
            status_txt = "NO FACE DETECTED"
            status_color = (0, 0, 220)
            vision_state["status_text"] = "No Face"
            # Fire event in a daemon thread to avoid blocking the detection loop
            threading.Thread(
                target=_notify_proctor_event,
                args=("Face Not Detected", 91),
                daemon=True,
            ).start()
        elif face_count > 1:
            status_txt = f"MULTIPLE FACES: {face_count}"
            status_color = (0, 160, 255)
            vision_state["status_text"] = f"Multiple Faces ({face_count})"
            threading.Thread(
                target=_notify_proctor_event,
                args=("Multiple Faces", min(99, 80 + face_count * 5)),
                daemon=True,
            ).start()
        else:
            status_txt = "Face OK"
            status_color = (0, 200, 80)
            vision_state["status_text"] = "Face OK"

        if phone_detected:
            vision_state["phone_detected"] = True
            cv2.putText(
                frame,
                f"PHONE ({phone_conf}%)",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 50, 200),
                2,
            )
            threading.Thread(
                target=_notify_proctor_event,
                args=("Phone Detected", phone_conf),
                daemon=True,
            ).start()
        else:
            vision_state["phone_detected"] = False

        vision_state["face_count"] = face_count

        # ── Draw annotation overlay ──
        # Dark status bar at top
        cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(frame, status_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # Candidate name top-right
        name = vision_state["student_name"]
        if name and name != "Unknown":
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, name, (w - tw - 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Red border when flagged
        if face_count == 0 or face_count > 1 or phone_detected:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 200), 4)

        # Timestamp bottom-left
        cv2.putText(
            frame,
            datetime.now().strftime("%H:%M:%S"),
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

        # Encode annotated frame to JPEG and store for MJPEG stream
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with vision_lock:
            vision_state["output_frame"] = jpeg.tobytes()

        time.sleep(0.05)  # ~20 fps


def _ensure_detection_thread() -> None:
    """Start the background detection thread exactly once (thread-safe)."""
    global _detection_thread_started
    with _detection_thread_lock:
        if not _detection_thread_started:
            _detection_thread_started = True
            t = threading.Thread(target=_detection_loop, daemon=True)
            t.start()


@app.before_request
def before_request() -> None:
    """Guarantee the detection thread is running before the first request."""
    _ensure_detection_thread()


# ─── Vision routes ────────────────────────────────────────────

@app.route("/proctor/frame/<app_number>", methods=["POST"])
def receive_frame(app_number: str):
    """
    Receive a JPEG frame from the candidate's browser (~5 fps).

    Accepts two content types:
      - application/json  → { "image": "<base64 data-URI or raw base64>" }
      - application/octet-stream → raw JPEG bytes
    """
    try:
        ct = request.content_type or ""
        if "json" in ct:
            data = request.get_json(force=True) or {}
            img_b64: str = data.get("image", "")
            # Strip data-URI prefix if present ("data:image/jpeg;base64,...")
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            raw = base64.b64decode(img_b64)
        else:
            raw = request.get_data()

        if not raw:
            return jsonify({"ok": False, "error": "empty body"}), 400

        with frames_lock:
            candidate_frames[app_number] = {
                "raw_jpeg": raw,
                "last_seen": time.time(),
            }

        # Auto-assign the first candidate to send frames as the active one
        if vision_state["app_number"] is None:
            vision_state["app_number"] = app_number

        return jsonify({"ok": True}), 200

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _generate_mjpeg():
    """Generator that yields annotated frames as an MJPEG multipart stream."""
    while True:
        with vision_lock:
            frame = vision_state.get("output_frame")
        if not frame:
            frame = _make_placeholder()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.route("/proctor/video_feed")
def video_feed():
    """MJPEG stream endpoint — embedded in the teacher dashboard as <img src>."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/proctor/set_candidate", methods=["POST"])
def proctor_set_candidate():
    """Tell the vision engine which candidate to monitor."""
    d = request.get_json(force=True) or {}
    vision_state["app_number"] = d.get("app_number")
    vision_state["student_name"] = d.get("student_name", "Unknown")
    vision_state["last_flag"] = {}  # reset cooldowns for the new session
    return jsonify({"status": "ok"})


@app.route("/proctor/clear_candidate", methods=["POST"])
def proctor_clear_candidate():
    """Stop monitoring (e.g. when the candidate closes their browser)."""
    vision_state["app_number"] = None
    vision_state["student_name"] = "Unknown"
    return jsonify({"status": "ok"})


@app.route("/proctor/status")
def proctor_status():
    """Return current detection state for the teacher live-feed widget."""
    return jsonify({
        "app_number": vision_state["app_number"],
        "student_name": vision_state["student_name"],
        "face_count": vision_state["face_count"],
        "phone_detected": vision_state["phone_detected"],
        "status_text": vision_state["status_text"],
    })


@app.route("/proctor/ping")
def proctor_ping():
    """Health-check for the vision subsystem."""
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — EXAM PORTAL
#  Candidate login, session management, answer saving,
#  teacher login, proctoring event log, snapshot storage.
# ═══════════════════════════════════════════════════════════════

# ─── Serve frontend ──────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "connected_portal.html")


# ─── Auth helpers ─────────────────────────────────────────────

def _hash(password: str) -> str:
    """Return a SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode()).hexdigest()


# ── Static credential stores (replace with a DB in production) ──

CANDIDATES: dict[str, dict] = {
    "240110012345": {"password": _hash("Pass@1234"), "name": "Arjun Mehta"},
    "240110056789": {"password": _hash("Pass@5678"), "name": "Priya Sharma"},
    "240110099001": {"password": _hash("Pass@9900"), "name": "Rahul Singh"},
}

TEACHERS: dict[str, str] = {
    "teacher1": _hash("Teacher@123"),
    "admin": _hash("Admin@2026"),
}

# ─── In-memory data stores ────────────────────────────────────

# { app_number: { name, logged_in_at, started, submitted, answers, ... } }
CANDIDATE_SESSIONS: dict[str, dict] = {}

# Flat list of violation events, newest first
PROCTOR_EVENTS: list[dict] = []

# Snapshot images (base64 data-URIs) captured on violation
SNAPSHOTS: list[dict] = []

# ─── Exam configuration ───────────────────────────────────────

EXAM_CONFIG: dict = {
    "title": "JEE (Main) 2026",
    "paper": "Paper 1 (B.E. / B.Tech)",
    "duration_mins": 180,
    "total_questions": 75,
    "total_marks": 300,
}

# ─── Event metadata: display level + risk impact ─────────────

EVENT_META: dict[str, dict] = {
    "Tab Switch":        {"level": "warn",   "impact": 20},
    "Face Not Detected": {"level": "danger", "impact": 25},
    "Multiple Faces":    {"level": "danger", "impact": 30},
    "Audio Detected":    {"level": "warn",   "impact": 10},
    "Browser Blur":      {"level": "warn",   "impact": 15},
    "Phone Detected":    {"level": "danger", "impact": 35},
    # Informational events — no risk impact
    "Exam Started":      {"level": "info",   "impact": 0},
    "Exam Submitted":    {"level": "info",   "impact": 0},
    "Candidate Login":   {"level": "info",   "impact": 0},
}

# ─── Response helpers ─────────────────────────────────────────

def _ok(data: dict | None = None, msg: str = "OK"):
    """Return a 200 JSON success response, optionally merging extra data."""
    body = {"status": "success", "message": msg}
    if data:
        body.update(data)
    return jsonify(body), 200


def _err(msg: str, code: int = 400):
    """Return an error JSON response with the given HTTP status code."""
    return jsonify({"status": "error", "message": msg}), code


def _now_str() -> str:
    """Current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")


def _now_iso() -> str:
    """Current time as ISO-8601 string."""
    return datetime.now().isoformat()


def _time_left(app_no: str) -> int:
    """Return seconds remaining in the exam, or 0 if not started / submitted."""
    sess = CANDIDATE_SESSIONS.get(app_no, {})
    if not sess.get("started") or sess.get("submitted"):
        return 0
    return max(0, int(sess["end_time"] - time.time()))


def log_event(
    app_no: str,
    event_type: str,
    confidence: int,
    impact_override: int | None = None,
    note: str = "",
) -> None:
    """
    Append a proctoring event to PROCTOR_EVENTS (newest first).

    impact_override lets callers (e.g. the browser client via /api/proctor/event)
    supply a custom risk impact instead of using the default from EVENT_META.
    """
    meta = EVENT_META.get(event_type, {"level": "danger", "impact": 10})
    impact = impact_override if impact_override is not None else meta["impact"]
    cand = CANDIDATES.get(app_no, {})
    PROCTOR_EVENTS.insert(0, {
        "id": str(uuid.uuid4()),
        "time": _now_str(),
        "timestamp": _now_iso(),
        "app_number": app_no,
        "student_name": cand.get("name", app_no),
        "event_type": event_type,
        "level": meta["level"],
        "confidence": confidence,
        "impact": impact,
        "note": note,
    })


def _risk_score(app_no: str) -> int:
    """Sum the impact of all non-info events for a candidate, capped at 100."""
    return min(
        100,
        sum(
            e["impact"]
            for e in PROCTOR_EVENTS
            if e["app_number"] == app_no and e["level"] != "info"
        ),
    )


def _events_for(app_no: str) -> list[dict]:
    """Return all non-info proctoring events for a specific candidate."""
    return [e for e in PROCTOR_EVENTS if e["app_number"] == app_no and e["level"] != "info"]


# ═══════════════════════════════════════════════════════════════
#  CANDIDATE ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/login", methods=["POST"])
def candidate_login():
    """Authenticate a candidate by application number + password."""
    d = request.get_json(force=True) or {}
    app_no = str(d.get("app_number", "")).strip()
    pw = str(d.get("password", "")).strip()

    if not app_no or not pw:
        return _err("Application number and password are required.")

    cand = CANDIDATES.get(app_no)
    if not cand or cand["password"] != _hash(pw):
        return _err("Invalid application number or password.", 401)

    # Prevent re-login after the exam has been submitted
    if CANDIDATE_SESSIONS.get(app_no, {}).get("submitted"):
        return _err("Your exam has already been submitted.", 403)

    session["app_number"] = app_no
    session["role"] = "candidate"

    # Create session record only on first login (allow page refresh without reset)
    if app_no not in CANDIDATE_SESSIONS:
        CANDIDATE_SESSIONS[app_no] = {
            "name": cand["name"],
            "logged_in_at": _now_iso(),
            "started": False,
            "submitted": False,
            "start_time": None,
            "end_time": None,
            "answers": {},
        }
        log_event(app_no, "Candidate Login", 100)

    return _ok(
        {
            "candidate": {"app_number": app_no, "name": cand["name"]},
            "exam": EXAM_CONFIG,
        },
        "Login successful",
    )


@app.route("/api/logout", methods=["GET", "POST"])
def candidate_logout():
    """Clear the candidate's server-side session."""
    session.clear()
    return _ok(msg="Logged out")


@app.route("/api/session", methods=["GET"])
def check_session():
    """Return the current candidate's session state (used on page load)."""
    if "app_number" not in session:
        return _err("No active session.", 401)

    app_no = session["app_number"]
    cand = CANDIDATES.get(app_no, {})
    sess = CANDIDATE_SESSIONS.get(app_no, {})

    return _ok({
        "app_number": app_no,
        "name": cand.get("name", "Unknown"),
        "exam_started": sess.get("started", False),
        "exam_submitted": sess.get("submitted", False),
        "time_remaining": _time_left(app_no),
    })


# ─── Exam lifecycle ───────────────────────────────────────────

@app.route("/api/exam/info", methods=["GET"])
def exam_info():
    """Return static exam configuration (title, duration, marks, etc.)."""
    return _ok({"exam": EXAM_CONFIG})


@app.route("/api/exam/start", methods=["POST"])
def exam_start():
    """
    Start the timed exam for an authenticated candidate.

    The candidate must have agreed to the instructions (agreed=true).
    Calling this endpoint a second time returns the remaining time
    instead of resetting the clock.
    """
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    d = request.get_json(force=True) or {}

    if not d.get("agreed"):
        return _err("You must agree to the instructions.")

    sess = CANDIDATE_SESSIONS.get(app_no, {})
    if sess.get("started"):
        # Idempotent — return remaining time so the frontend can sync
        return _ok(
            {"already_started": True, "time_remaining": _time_left(app_no)},
            "Exam already in progress",
        )

    ts = time.time()
    CANDIDATE_SESSIONS[app_no].update({
        "started": True,
        "start_time": ts,
        "end_time": ts + EXAM_CONFIG["duration_mins"] * 60,
    })
    log_event(app_no, "Exam Started", 100)

    return _ok(
        {
            "start_time": ts,
            "end_time": CANDIDATE_SESSIONS[app_no]["end_time"],
            "duration_mins": EXAM_CONFIG["duration_mins"],
        },
        "Exam started",
    )


@app.route("/api/exam/answer", methods=["POST"])
def save_answer():
    """
    Save or update a single answer.

    Body: { "question_id": str, "answer": any }
    Answers are stored in the session dict and overwrite any previous value.
    """
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    sess = CANDIDATE_SESSIONS.get(app_no, {})

    if not sess.get("started"):
        return _err("Exam not started.")
    if sess.get("submitted"):
        return _err("Exam already submitted.")
    if time.time() > sess["end_time"]:
        return _err("Time is up.")

    d = request.get_json(force=True) or {}
    qid = d.get("question_id")
    if not qid:
        return _err("question_id is required.")

    sess["answers"][qid] = d.get("answer")
    return _ok({"saved": qid})


@app.route("/api/exam/submit", methods=["POST"])
def exam_submit():
    """
    Submit the exam.

    Accepts a final bulk answers dict in the body so the frontend can
    flush any unsaved answers atomically on submit.
    Body: { "answers": { question_id: answer, ... } }
    """
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    sess = CANDIDATE_SESSIONS.get(app_no)

    if not sess or not sess.get("started"):
        return _err("Exam not started.")
    if sess.get("submitted"):
        return _err("Exam already submitted.")

    d = request.get_json(force=True) or {}
    # Merge any final answers sent with the submit request
    sess["answers"].update(d.get("answers", {}))
    sess["submitted"] = True
    sess["submit_time"] = time.time()

    log_event(app_no, "Exam Submitted", 100)
    return _ok({"total_answered": len(sess["answers"])}, "Exam submitted successfully")


@app.route("/api/exam/status", methods=["GET"])
def exam_status():
    """Return real-time exam progress for the authenticated candidate."""
    if "app_number" not in session:
        return _err("Not authenticated.", 401)

    app_no = session["app_number"]
    sess = CANDIDATE_SESSIONS.get(app_no, {})

    return _ok({
        "started": sess.get("started", False),
        "submitted": sess.get("submitted", False),
        "time_remaining": _time_left(app_no),
        "answered_count": len(sess.get("answers", {})),
    })


# ═══════════════════════════════════════════════════════════════
#  TEACHER ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/teacher/login", methods=["POST"])
def teacher_login():
    """Authenticate a teacher by username + password."""
    d = request.get_json(force=True) or {}
    user = str(d.get("username", "")).strip()
    pw = str(d.get("password", "")).strip()

    if not user or not pw:
        return _err("Username and password required.")

    stored = TEACHERS.get(user)
    if not stored or stored != _hash(pw):
        return _err("Invalid credentials.", 401)

    session["teacher"] = user
    session["role"] = "teacher"
    return _ok({"username": user}, "Teacher login successful")


@app.route("/api/teacher/logout", methods=["POST"])
def teacher_logout():
    """Clear the teacher's server-side session."""
    session.clear()
    return _ok(msg="Logged out")


@app.route("/api/teacher/students", methods=["GET"])
def teacher_students():
    """
    Return a list of all candidates with their live risk scores.

    Sorted by risk score descending so the most suspicious candidates
    appear first in the teacher dashboard.
    """
    students = []
    for app_no, sess in CANDIDATE_SESSIONS.items():
        evts = _events_for(app_no)
        risk = _risk_score(app_no)
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
            "risk_score": risk,
            "event_count": len(evts),
            "last_event": status_label,
            "time_remaining": _time_left(app_no),
            "answered_count": len(sess.get("answers", {})),
        })

    students.sort(key=lambda s: s["risk_score"], reverse=True)
    return _ok({"students": students, "total": len(students)})


# ═══════════════════════════════════════════════════════════════
#  PROCTORING EVENT ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/proctor/event", methods=["POST"])
def proctor_log():
    """
    Receive a proctoring event from the browser-side detection engine.

    The browser sends events for: Tab Switch, Browser Blur, Audio Detected,
    Face Not Detected, Multiple Faces (via face-api.js).
    The server-side vision engine writes events directly via log_event().
    """
    d = request.get_json(force=True) or {}
    app_no = d.get("app_number", "UNKNOWN")
    event_type = d.get("event_type", "Unknown")
    confidence = int(d.get("confidence", 0))
    note = d.get("note", "")
    # Allow the browser to override the default risk impact
    impact_override = int(d["impact"]) if "impact" in d else None

    log_event(app_no, event_type, confidence, impact_override, note=note)
    return _ok(msg="Event logged")


@app.route("/api/proctor/events", methods=["GET"])
def proctor_events():
    """
    Return proctoring events, optionally filtered by student name or event type.

    Query params:
      student — partial case-insensitive name match
      type    — exact event type match
    """
    student_filter = request.args.get("student", "").lower()
    type_filter = request.args.get("type", "")

    filtered = [
        e for e in PROCTOR_EVENTS
        if (not student_filter or student_filter in e["student_name"].lower())
        and (not type_filter or e["event_type"] == type_filter)
    ]
    return _ok({"events": filtered, "total": len(filtered)})


@app.route("/api/proctor/summary", methods=["GET"])
def proctor_summary():
    """
    Return aggregate statistics for the teacher dashboard stat bar.

    flagged  — candidates with risk score ≥ 70
    warnings — candidates with risk score 30–69
    """
    total = len(CANDIDATE_SESSIONS)
    active = sum(
        1 for s in CANDIDATE_SESSIONS.values()
        if s.get("started") and not s.get("submitted")
    )
    submitted = sum(1 for s in CANDIDATE_SESSIONS.values() if s.get("submitted"))

    # Build per-candidate risk totals from the event log
    risk_map: dict[str, int] = {}
    for e in PROCTOR_EVENTS:
        if e["level"] == "info":
            continue
        risk_map[e["app_number"]] = min(
            100, risk_map.get(e["app_number"], 0) + e["impact"]
        )

    return _ok({
        "logged_in": total,
        "active": active,
        "waiting": total - active - submitted,
        "submitted": submitted,
        "flagged": sum(1 for v in risk_map.values() if v >= 70),
        "warnings": sum(1 for v in risk_map.values() if 30 <= v < 70),
        "total_events": sum(1 for e in PROCTOR_EVENTS if e["level"] != "info"),
    })


@app.route("/api/proctor/export", methods=["GET"])
def proctor_export():
    """Export the non-info proctoring event log as a CSV download."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Time", "App Number", "Student", "Event", "Confidence", "Impact", "Level", "Note"])
    for e in PROCTOR_EVENTS:
        if e["level"] == "info":
            continue
        writer.writerow([
            e["time"],
            e["app_number"],
            e["student_name"],
            e["event_type"],
            f"{e['confidence']}%",
            f"+{e['impact']}",
            e["level"],
            e.get("note", ""),
        ])
    out.seek(0)
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit-report.csv"},
    )


@app.route("/api/proctor/snapshot", methods=["POST"])
def proctor_snapshot():
    """
    Store a base64 JPEG snapshot captured by the browser on a violation.

    Keeps a maximum of 500 snapshots (oldest discarded).
    Body: { app_number, student_name, event_type, image: "data:image/jpeg;base64,..." }
    """
    d = request.get_json(force=True) or {}
    image = d.get("image", "")

    if not image or not image.startswith("data:image"):
        return _err("Invalid image data.")

    # Discard oldest snapshot if we're at capacity
    if len(SNAPSHOTS) >= 500:
        SNAPSHOTS.pop()

    SNAPSHOTS.insert(0, {
        "id": str(uuid.uuid4()),
        "time": _now_str(),
        "timestamp": _now_iso(),
        "app_number": d.get("app_number", ""),
        "student_name": d.get("student_name", ""),
        "event_type": d.get("event_type", ""),
        "image": image,
    })
    return _ok(msg="Snapshot saved")


@app.route("/api/proctor/snapshots", methods=["GET"])
def proctor_snapshots():
    """
    Return stored snapshots, optionally filtered by student name or event type.

    Query params:
      student — partial case-insensitive name match
      type    — exact event type match
    """
    student_filter = request.args.get("student", "").lower()
    type_filter = request.args.get("type", "")

    filtered = [
        s for s in SNAPSHOTS
        if (not student_filter or student_filter in s["student_name"].lower())
        and (not type_filter or s["event_type"] == type_filter)
    ]
    return _ok({"snapshots": filtered, "total": len(filtered)})


# ─── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*52}")
    print(f"  Open → http://localhost:{port}")
    print(f"{'='*52}")
    print("  CANDIDATE: 240110012345 / Pass@1234")
    print("  TEACHER:   teacher1 / Teacher@123")
    print(f"{'='*52}\n")
    # debug=False is intentional — debug mode restarts the process
    # which would kill the detection thread
    app.run(debug=False, host="0.0.0.0", port=port)
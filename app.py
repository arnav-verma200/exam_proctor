"""
=============================================================
  Exam Portal — Flask Backend (app.py)

  HOW TO RUN:
  1. Put app.py + connected_portal.html in the SAME folder
  2. pip install flask flask-cors
  3. python app.py
  4. Open browser → http://localhost:5000
=============================================================
"""

from flask import Flask, request, jsonify, session, Response, send_from_directory
from flask_cors import CORS
from datetime import datetime
import csv, io, hashlib, uuid, time, os

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "exam_portal_secret_2026")

# In production the frontend and backend share the same origin,
# so CORS is only needed for explicit cross-origin callers (e.g. proctor.py).
# Pull allowed origins from an env var; fall back to localhost for local dev.
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5000,http://127.0.0.1:5000"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)

# Secure session cookies in production
IS_PRODUCTION = os.environ.get("FLASK_ENV", "development") == "production"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,   # requires HTTPS in prod
)

# ─────────────────────────────────────────
#  OPTIONAL: PROXY /video_feed to proctor.py
#  Allows the browser to use a single origin for both app and proctor.
# ─────────────────────────────────────────
PROCTOR_URL = os.environ.get("PROCTOR_URL", "http://localhost:5001")

@app.route("/video_feed")
def video_feed_proxy():
    """Proxy MJPEG stream from proctor.py so front-end stays same-origin."""
    import urllib.request
    try:
        upstream = urllib.request.urlopen(PROCTOR_URL + "/video_feed", timeout=5)
        def generate():
            try:
                while True:
                    chunk = upstream.read(4096)
                    if not chunk:
                        break
                    yield chunk
            finally:
                upstream.close()
        return Response(generate(),
                        content_type=upstream.headers.get("Content-Type",
                                                          "multipart/x-mixed-replace; boundary=frame"))
    except Exception:
        # Proctor offline — return a 1×1 grey JPEG placeholder
        import base64
        grey = base64.b64decode(
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
            "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAARC"
            "AABAAEDASIA"
        )
        return Response(grey, content_type="image/jpeg")



@app.route("/")
def index():
    return send_from_directory("frontend", "connected_portal.html")

# ─────────────────────────────────────────
#  IN-MEMORY DATA STORE
# ─────────────────────────────────────────

def h(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

CANDIDATES = {
    "240110012345": {"password": h("Pass@1234"), "name": "Arjun Mehta"},
    "240110056789": {"password": h("Pass@5678"), "name": "Priya Sharma"},
    "240110099001": {"password": h("Pass@9900"), "name": "Rahul Singh"},
}

TEACHERS = {
    "teacher1": h("Teacher@123"),
    "admin":    h("Admin@2026"),
}

# Tracks every candidate who has logged in (even before starting exam)
# { app_number: { name, logged_in_at, started, submitted, start_time, end_time, answers } }
CANDIDATE_SESSIONS = {}

# All proctoring events in chronological order (newest first after insert)
PROCTOR_EVENTS = []
CLIPS          = []   # evidence clips saved to Appwrite

EXAM_CONFIG = {
    "title":           "JEE (Main) 2026",
    "paper":           "Paper 1 (B.E. / B.Tech)",
    "duration_mins":   180,
    "total_questions": 75,
    "total_marks":     300,
}

# Event type metadata — used to set correct level and impact if not provided
EVENT_META = {
    "Tab Switch":        {"level": "danger", "impact": 20},
    "Face Not Detected": {"level": "danger", "impact": 25},
    "Multiple Faces":    {"level": "danger", "impact": 30},
    "Audio Detected":    {"level": "warn",   "impact": 10},
    "Browser Blur":      {"level": "warn",   "impact": 15},
    "Phone Detected":    {"level": "danger", "impact": 35},
    "Exam Started":      {"level": "info",   "impact": 0},
    "Exam Submitted":    {"level": "info",   "impact": 0},
    "Candidate Login":   {"level": "info",   "impact": 0},
    "Tab Switch":        {"level": "warn",   "impact": 20},
    "Browser Blur":      {"level": "warn",   "impact": 15},
}

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def ok(data=None, msg="OK"):
    body = {"status": "success", "message": msg}
    if data:
        body.update(data)
    return jsonify(body), 200

def err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code

def now_str():
    return datetime.now().strftime("%H:%M:%S")

def now_iso():
    return datetime.now().isoformat()

def time_left(app_no):
    s = CANDIDATE_SESSIONS.get(app_no, {})
    if not s.get("started") or s.get("submitted"):
        return 0
    return max(0, int(s["end_time"] - time.time()))

def log_event(app_no, event_type, confidence, impact_override=None, note=""):
    """Log a real proctoring event. Level and impact come from EVENT_META unless overridden."""
    meta   = EVENT_META.get(event_type, {"level": "danger", "impact": 10})
    level  = meta["level"]
    impact = impact_override if impact_override is not None else meta["impact"]
    cand   = CANDIDATES.get(app_no, {})

    PROCTOR_EVENTS.insert(0, {   # insert at front → newest first
        "id":           str(uuid.uuid4()),
        "time":         now_str(),
        "timestamp":    now_iso(),
        "app_number":   app_no,
        "student_name": cand.get("name", app_no),
        "event_type":   event_type,
        "level":        level,
        "confidence":   confidence,
        "impact":       impact,
        "note":         note,
    })

def risk_score_for(app_no):
    """Calculate cumulative risk score for a candidate (capped at 100)."""
    return min(100, sum(
        e["impact"] for e in PROCTOR_EVENTS
        if e["app_number"] == app_no and e["level"] != "info"
    ))

def events_for(app_no):
    """All non-info proctoring events for a candidate."""
    return [e for e in PROCTOR_EVENTS
            if e["app_number"] == app_no and e["level"] != "info"]

# ─────────────────────────────────────────
#  CANDIDATE AUTH
# ─────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def candidate_login():
    d      = request.get_json(force=True) or {}
    app_no = str(d.get("app_number", "")).strip()
    pw     = str(d.get("password", "")).strip()

    if not app_no or not pw:
        return err("Application number and password are required.")

    cand = CANDIDATES.get(app_no)
    if not cand or cand["password"] != h(pw):
        return err("Invalid application number or password.", 401)

    if CANDIDATE_SESSIONS.get(app_no, {}).get("submitted"):
        return err("Your exam has already been submitted.", 403)

    # Register session
    session["app_number"] = app_no
    session["role"]       = "candidate"

    # Track in global store so teacher can see the student immediately
    if app_no not in CANDIDATE_SESSIONS:
        CANDIDATE_SESSIONS[app_no] = {
            "name":          cand["name"],
            "logged_in_at":  now_iso(),
            "started":       False,
            "submitted":     False,
            "start_time":    None,
            "end_time":      None,
            "answers":       {},
        }
        log_event(app_no, "Candidate Login", 100, note="")

    return ok({
        "candidate": {"app_number": app_no, "name": cand["name"]},
        "exam": EXAM_CONFIG,
    }, "Login successful")


@app.route("/api/logout", methods=["GET", "POST"])
def candidate_logout():
    session.clear()
    return ok(msg="Logged out")


@app.route("/api/session", methods=["GET"])
def check_session():
    if "app_number" not in session:
        return err("No active session.", 401)
    app_no = session["app_number"]
    cand   = CANDIDATES.get(app_no, {})
    sess   = CANDIDATE_SESSIONS.get(app_no, {})
    return ok({
        "app_number":     app_no,
        "name":           cand.get("name", "Unknown"),
        "exam_started":   sess.get("started", False),
        "exam_submitted": sess.get("submitted", False),
        "time_remaining": time_left(app_no),
    })

# ─────────────────────────────────────────
#  EXAM ROUTES
# ─────────────────────────────────────────

@app.route("/api/exam/info", methods=["GET"])
def exam_info():
    return ok({"exam": EXAM_CONFIG})


@app.route("/api/exam/start", methods=["POST"])
def exam_start():
    if "app_number" not in session:
        return err("Not authenticated.", 401)

    d      = request.get_json(force=True) or {}
    app_no = session["app_number"]

    if not d.get("agreed"):
        return err("You must agree to the instructions.")

    sess = CANDIDATE_SESSIONS.get(app_no, {})
    if sess.get("started"):
        return ok({"already_started": True, "time_remaining": time_left(app_no)},
                  "Exam already in progress")

    ts = time.time()
    CANDIDATE_SESSIONS[app_no].update({
        "started":    True,
        "start_time": ts,
        "end_time":   ts + EXAM_CONFIG["duration_mins"] * 60,
    })

    log_event(app_no, "Exam Started", 100)
    return ok({
        "start_time":    ts,
        "end_time":      CANDIDATE_SESSIONS[app_no]["end_time"],
        "duration_mins": EXAM_CONFIG["duration_mins"],
    }, "Exam started")


@app.route("/api/exam/answer", methods=["POST"])
def save_answer():
    if "app_number" not in session:
        return err("Not authenticated.", 401)
    app_no = session["app_number"]
    sess   = CANDIDATE_SESSIONS.get(app_no, {})
    if not sess.get("started"):            return err("Exam not started.")
    if sess.get("submitted"):              return err("Already submitted.")
    if time.time() > sess["end_time"]:     return err("Time is up.")
    d = request.get_json(force=True) or {}
    qid = d.get("question_id")
    if not qid:                            return err("question_id required.")
    sess["answers"][qid] = d.get("answer")
    return ok({"saved": qid})


@app.route("/api/exam/submit", methods=["POST"])
def exam_submit():
    if "app_number" not in session:
        return err("Not authenticated.", 401)
    app_no = session["app_number"]
    sess   = CANDIDATE_SESSIONS.get(app_no)
    if not sess or not sess.get("started"): return err("Exam not started.")
    if sess.get("submitted"):               return err("Already submitted.")
    d = request.get_json(force=True) or {}
    sess["answers"].update(d.get("answers", {}))
    sess["submitted"]   = True
    sess["submit_time"] = time.time()
    log_event(app_no, "Exam Submitted", 100)
    return ok({"total_answered": len(sess["answers"])}, "Exam submitted successfully")


@app.route("/api/exam/status", methods=["GET"])
def exam_status():
    if "app_number" not in session:
        return err("Not authenticated.", 401)
    app_no = session["app_number"]
    sess   = CANDIDATE_SESSIONS.get(app_no, {})
    return ok({
        "started":        sess.get("started", False),
        "submitted":      sess.get("submitted", False),
        "time_remaining": time_left(app_no),
        "answered_count": len(sess.get("answers", {})),
    })

# ─────────────────────────────────────────
#  TEACHER AUTH
# ─────────────────────────────────────────

@app.route("/api/teacher/login", methods=["POST"])
def teacher_login():
    d    = request.get_json(force=True) or {}
    user = str(d.get("username", "")).strip()
    pw   = str(d.get("password", "")).strip()
    if not user or not pw:
        return err("Username and password required.")
    stored = TEACHERS.get(user)
    if not stored or stored != h(pw):
        return err("Invalid credentials.", 401)
    session["teacher"] = user
    session["role"]    = "teacher"
    return ok({"username": user}, "Teacher login successful")


@app.route("/api/teacher/logout", methods=["POST"])
def teacher_logout():
    session.clear()
    return ok(msg="Logged out")

# ─────────────────────────────────────────
#  TEACHER DASHBOARD — STUDENT LIST
# ─────────────────────────────────────────

@app.route("/api/teacher/students", methods=["GET"])
def teacher_students():
    students = []
    for app_no, sess in CANDIDATE_SESSIONS.items():
        evts       = events_for(app_no)   # only non-info events
        risk       = risk_score_for(app_no)
        last_evt   = evts[0]["event_type"] if evts else "Logged In"  # evts is newest-first

        # Status label shown on card
        if sess.get("submitted"):
            status_label = "Submitted"
        elif sess.get("started"):
            status_label = last_evt
        else:
            status_label = "Waiting"

        students.append({
            "app_number":     app_no,
            "name":           sess["name"],
            "started":        sess.get("started", False),
            "submitted":      sess.get("submitted", False),
            "risk_score":     risk,
            "event_count":    len(evts),
            "last_event":     status_label,
            "time_remaining": time_left(app_no),
            "answered_count": len(sess.get("answers", {})),
        })

    # Sort by risk descending so highest-risk students appear first
    students.sort(key=lambda s: s["risk_score"], reverse=True)
    return ok({"students": students, "total": len(students)})

# ─────────────────────────────────────────
#  PROCTORING — EVENTS
# ─────────────────────────────────────────

@app.route("/api/proctor/event", methods=["POST"])
def proctor_log():
    """Receives a real proctoring event from browser OR proctor.py (no session required)."""
    d          = request.get_json(force=True) or {}
    app_no     = d.get("app_number", "UNKNOWN")
    event_type = d.get("event_type", "Unknown")
    confidence = int(d.get("confidence", 0))
    note       = d.get("note", "")

    impact_override = int(d["impact"]) if "impact" in d else None
    log_event(app_no, event_type, confidence, impact_override, note=note)
    return ok(msg="Event logged")


@app.route("/api/proctor/events", methods=["GET"])
def proctor_events():
    """Returns all proctoring events, newest first. Supports ?student= and ?type= filters."""
    sf   = request.args.get("student", "").lower()
    tf   = request.args.get("type", "")

    evts = [
        e for e in PROCTOR_EVENTS
        if (not sf or sf in e["student_name"].lower())
        and (not tf or e["event_type"] == tf)
    ]
    return ok({"events": evts, "total": len(evts)})


@app.route("/api/proctor/summary", methods=["GET"])
def proctor_summary():
    """Stats bar numbers for the teacher dashboard."""
    total_logged_in = len(CANDIDATE_SESSIONS)
    total_active    = sum(1 for s in CANDIDATE_SESSIONS.values()
                         if s.get("started") and not s.get("submitted"))
    total_submitted = sum(1 for s in CANDIDATE_SESSIONS.values()
                         if s.get("submitted"))
    total_waiting   = sum(1 for s in CANDIDATE_SESSIONS.values()
                         if not s.get("started"))

    # Build per-student risk scores (ignore info events)
    risk_map = {}
    for e in PROCTOR_EVENTS:
        if e["level"] == "info":
            continue
        an = e["app_number"]
        risk_map[an] = min(100, risk_map.get(an, 0) + e["impact"])

    flagged  = sum(1 for v in risk_map.values() if v >= 70)
    warnings = sum(1 for v in risk_map.values() if 30 <= v < 70)

    # Total events excludes info-level (login/start/submit)
    real_events = sum(1 for e in PROCTOR_EVENTS if e["level"] != "info")

    return ok({
        "logged_in":    total_logged_in,
        "active":       total_active,
        "waiting":      total_waiting,
        "submitted":    total_submitted,
        "flagged":      flagged,
        "warnings":     warnings,
        "total_events": real_events,
    })


@app.route("/api/proctor/export", methods=["GET"])
def proctor_export():
    """Download audit log as CSV (excludes info events)."""
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["Time", "App Number", "Student", "Event", "Confidence", "Impact", "Level"])
    for e in PROCTOR_EVENTS:
        if e["level"] == "info":
            continue
        w.writerow([
            e["time"], e["app_number"], e["student_name"],
            e["event_type"], str(e["confidence"]) + "%",
            "+" + str(e["impact"]), e["level"],
        ])
    out.seek(0)
    return Response(
        out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit-report.csv"}
    )

@app.route("/api/proctor/clip", methods=["POST"])
def proctor_save_clip():
    """Called by student browser after uploading a clip to Appwrite."""
    d = request.get_json(force=True) or {}
    CLIPS.insert(0, {
        "id":           str(uuid.uuid4()),
        "time":         now_str(),
        "timestamp":    now_iso(),
        "app_number":   d.get("app_number",""),
        "student_name": d.get("student_name",""),
        "event_type":   d.get("event_type",""),
        "file_id":      d.get("file_id",""),
        "file_name":    d.get("file_name",""),
        "bucket_id":    d.get("bucket_id",""),
    })
    return ok(msg="Clip saved")


@app.route("/api/proctor/clips", methods=["GET"])
def proctor_get_clips():
    """Returns all saved evidence clips."""
    sf = request.args.get("student","").lower()
    tf = request.args.get("type","")
    clips = [c for c in CLIPS
             if (not sf or sf in c["student_name"].lower())
             and (not tf or c["event_type"]==tf)]
    return ok({"clips": clips, "total": len(clips)})


# ─────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 52)
    print(f"  ✅  Open → http://localhost:{port}")
    print("=" * 52)
    print("  CANDIDATE LOGIN")
    print("  240110012345 / Pass@1234  → Arjun Mehta")
    print("  240110056789 / Pass@5678  → Priya Sharma")
    print("  240110099001 / Pass@9900  → Rahul Singh")
    print("  TEACHER LOGIN")
    print("  teacher1 / Teacher@123")
    print("  admin    / Admin@2026")
    print("=" * 52 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
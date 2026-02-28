"""
=============================================================
  app.py  — Exam Portal + Proctor Vision Engine (MERGED)

  Everything in one process. Deploy to Render with one service.

  Browser pushes JPEG frames → POST /proctor/frame/<app_number>
  Server runs OpenCV + MediaPipe face detection on those frames
  Teacher sees annotated MJPEG stream → GET /proctor/video_feed
  Violations POSTed internally to /api/proctor/event

  Run locally:   python app.py
  Deploy:        gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
                 (workers=1 is REQUIRED — detection state is in-process memory)
=============================================================
"""

from flask import Flask, request, jsonify, session, Response, send_from_directory
from flask_cors import CORS
from datetime import datetime
import csv, io, hashlib, uuid, time, os, threading, base64

# ── OpenCV + MediaPipe (installed via requirements.txt) ───────
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "exam_portal_secret_2026")

_raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]
CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)

IS_PRODUCTION = os.environ.get("FLASK_ENV", "development") == "production"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
)

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — VISION ENGINE
#  Browser sends JPEG frames → we run MediaPipe → serve MJPEG
# ═══════════════════════════════════════════════════════════════

# MediaPipe face detector
_mp_face = mp.solutions.face_detection
face_detector = _mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

VISION_COOLDOWN = {"Face Not Detected": 8, "Multiple Faces": 8, "Phone Detected": 10}
FRAME_TIMEOUT   = 10   # seconds — if no frame received, show placeholder

# One raw frame slot per candidate
candidate_frames = {}   # { app_number: { raw_jpeg: bytes, last_seen: float } }
frames_lock      = threading.Lock()

vision_state = {
    "app_number":     None,
    "student_name":   "Unknown",
    "face_count":     0,
    "phone_detected": False,
    "status_text":    "Waiting for candidate…",
    "last_flag":      {},
    "output_frame":   None,   # annotated JPEG bytes for MJPEG
}
vision_lock = threading.Lock()


def _make_placeholder(text="Waiting for candidate…"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (130, 130, 130), 2)
    cv2.putText(img, datetime.now().strftime("%H:%M:%S"), (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes()


def _detect_phone(frame_bgr, face_bboxes):
    h, w  = frame_bgr.shape[:2]
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    for cnt in cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if not (0.03 * w * h < area < 0.35 * w * h):
            continue
        rect   = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 5 or rh < 5:
            continue
        if 1.5 <= max(rw, rh) / min(rw, rh) <= 2.5:
            cx, cy = int(rect[0][0]), int(rect[0][1])
            if not any(fx <= cx <= fx+fw and fy <= cy <= fy+fh for fx,fy,fw,fh in face_bboxes):
                return True, min(95, int(60 + (area/(w*h))*200))
    return False, 0


def _notify_proctor_event(event_type, confidence):
    """Log a vision event directly into the in-process event store."""
    app_no = vision_state["app_number"]
    if not app_no:
        return
    now  = time.time()
    last = vision_state["last_flag"].get(event_type, 0)
    if now - last < VISION_COOLDOWN.get(event_type, 5):
        return
    vision_state["last_flag"][event_type] = now
    log_event(app_no, event_type, confidence)
    print(f"[vision] ⚠  {event_type} ({confidence}%) → {app_no}")


def _detection_loop():
    """Background thread: pick up latest browser frame, run CV, store output."""
    while True:
        app_no   = vision_state["app_number"]
        raw_jpeg = None
        last_seen = 0

        if app_no:
            with frames_lock:
                slot = candidate_frames.get(app_no)
                if slot:
                    raw_jpeg  = slot["raw_jpeg"]
                    last_seen = slot["last_seen"]

        if raw_jpeg is None or (time.time() - last_seen) > FRAME_TIMEOUT:
            txt = f"Waiting for {vision_state['student_name']}…" if app_no else "No candidate active"
            with vision_lock:
                vision_state["output_frame"] = _make_placeholder(txt)
            time.sleep(0.1)
            continue

        try:
            arr   = np.frombuffer(raw_jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("imdecode failed")
        except Exception as e:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = frame.shape[:2]

        results    = face_detector.process(rgb)
        face_count = 0
        face_bboxes = []

        if results.detections:
            face_count = len(results.detections)
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x, y = max(0, int(bb.xmin*w)), max(0, int(bb.ymin*h))
                bw, bh = int(bb.width*w), int(bb.height*h)
                face_bboxes.append((x, y, bw, bh))
                col = (0, 220, 80) if face_count == 1 else (0, 80, 220)
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), col, 2)
                cv2.putText(frame, f"Face {int(det.score[0]*100)}%",
                            (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        phone_det, phone_conf = _detect_phone(frame, face_bboxes)

        if face_count == 0:
            txt = "NO FACE DETECTED"; col = (0, 0, 220)
            vision_state["status_text"] = "No Face"
            threading.Thread(target=_notify_proctor_event, args=("Face Not Detected", 91), daemon=True).start()
        elif face_count > 1:
            txt = f"MULTIPLE FACES: {face_count}"; col = (0, 160, 255)
            vision_state["status_text"] = f"Multiple Faces ({face_count})"
            threading.Thread(target=_notify_proctor_event, args=("Multiple Faces", min(99,80+face_count*5)), daemon=True).start()
        else:
            txt = "Face OK"; col = (0, 200, 80)
            vision_state["status_text"] = "Face OK"

        if phone_det:
            vision_state["phone_detected"] = True
            cv2.putText(frame, f"PHONE ({phone_conf}%)", (10, h-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 200), 2)
            threading.Thread(target=_notify_proctor_event, args=("Phone Detected", phone_conf), daemon=True).start()
        else:
            vision_state["phone_detected"] = False

        vision_state["face_count"] = face_count

        cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(frame, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        name = vision_state["student_name"]
        if name and name != "Unknown":
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, name, (w-tw-10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        if face_count == 0 or face_count > 1 or phone_det:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 200), 4)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with vision_lock:
            vision_state["output_frame"] = jpeg.tobytes()

        time.sleep(0.05)

# Start detection thread when the app loads
threading.Thread(target=_detection_loop, daemon=True).start()

# ─── Vision routes ────────────────────────────────────────────

@app.route("/proctor/frame/<app_number>", methods=["POST"])
def receive_frame(app_number):
    """Browser POSTs a JPEG frame here every ~200ms."""
    try:
        ct = request.content_type or ""
        if "json" in ct:
            data    = request.get_json(force=True) or {}
            img_b64 = data.get("image", "")
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            raw = base64.b64decode(img_b64)
        else:
            raw = request.get_data()
        if not raw:
            return jsonify({"ok": False, "error": "empty"}), 400
        with frames_lock:
            candidate_frames[app_number] = {"raw_jpeg": raw, "last_seen": time.time()}
        if vision_state["app_number"] is None:
            vision_state["app_number"] = app_number
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _generate_mjpeg():
    while True:
        with vision_lock:
            frame = vision_state.get("output_frame")
        if not frame:
            frame = _make_placeholder()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.route("/proctor/video_feed")
def video_feed():
    """MJPEG stream — teacher dashboard shows this."""
    return Response(_generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/proctor/set_candidate", methods=["POST"])
def proctor_set_candidate():
    d = request.get_json(force=True) or {}
    vision_state["app_number"]   = d.get("app_number")
    vision_state["student_name"] = d.get("student_name", "Unknown")
    vision_state["last_flag"]    = {}
    return jsonify({"status": "ok"})


@app.route("/proctor/clear_candidate", methods=["POST"])
def proctor_clear_candidate():
    vision_state["app_number"]   = None
    vision_state["student_name"] = "Unknown"
    return jsonify({"status": "ok"})


@app.route("/proctor/status")
def proctor_status():
    return jsonify({
        "app_number":     vision_state["app_number"],
        "student_name":   vision_state["student_name"],
        "face_count":     vision_state["face_count"],
        "phone_detected": vision_state["phone_detected"],
        "status_text":    vision_state["status_text"],
    })


@app.route("/proctor/ping")
def proctor_ping():
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — EXAM PORTAL (unchanged from original app.py)
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("frontend", "connected_portal.html")


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

CANDIDATE_SESSIONS = {}
PROCTOR_EVENTS     = []
SNAPSHOTS          = []

EXAM_CONFIG = {
    "title":           "JEE (Main) 2026",
    "paper":           "Paper 1 (B.E. / B.Tech)",
    "duration_mins":   180,
    "total_questions": 75,
    "total_marks":     300,
}

EVENT_META = {
    "Tab Switch":        {"level": "warn",   "impact": 20},
    "Face Not Detected": {"level": "danger", "impact": 25},
    "Multiple Faces":    {"level": "danger", "impact": 30},
    "Audio Detected":    {"level": "warn",   "impact": 10},
    "Browser Blur":      {"level": "warn",   "impact": 15},
    "Phone Detected":    {"level": "danger", "impact": 35},
    "Exam Started":      {"level": "info",   "impact": 0},
    "Exam Submitted":    {"level": "info",   "impact": 0},
    "Candidate Login":   {"level": "info",   "impact": 0},
}


def ok(data=None, msg="OK"):
    body = {"status": "success", "message": msg}
    if data:
        body.update(data)
    return jsonify(body), 200

def err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code

def now_str():  return datetime.now().strftime("%H:%M:%S")
def now_iso():  return datetime.now().isoformat()

def time_left(app_no):
    s = CANDIDATE_SESSIONS.get(app_no, {})
    if not s.get("started") or s.get("submitted"):
        return 0
    return max(0, int(s["end_time"] - time.time()))

def log_event(app_no, event_type, confidence, impact_override=None, note=""):
    meta   = EVENT_META.get(event_type, {"level": "danger", "impact": 10})
    impact = impact_override if impact_override is not None else meta["impact"]
    cand   = CANDIDATES.get(app_no, {})
    PROCTOR_EVENTS.insert(0, {
        "id":           str(uuid.uuid4()),
        "time":         now_str(),
        "timestamp":    now_iso(),
        "app_number":   app_no,
        "student_name": cand.get("name", app_no),
        "event_type":   event_type,
        "level":        meta["level"],
        "confidence":   confidence,
        "impact":       meta["impact"] if impact_override is None else impact_override,
        "note":         note,
    })

def risk_score_for(app_no):
    return min(100, sum(e["impact"] for e in PROCTOR_EVENTS
                        if e["app_number"] == app_no and e["level"] != "info"))

def events_for(app_no):
    return [e for e in PROCTOR_EVENTS
            if e["app_number"] == app_no and e["level"] != "info"]


# ── Candidate auth ────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def candidate_login():
    d      = request.get_json(force=True) or {}
    app_no = str(d.get("app_number","")).strip()
    pw     = str(d.get("password","")).strip()
    if not app_no or not pw:
        return err("Application number and password are required.")
    cand = CANDIDATES.get(app_no)
    if not cand or cand["password"] != h(pw):
        return err("Invalid application number or password.", 401)
    if CANDIDATE_SESSIONS.get(app_no, {}).get("submitted"):
        return err("Your exam has already been submitted.", 403)
    session["app_number"] = app_no
    session["role"]       = "candidate"
    if app_no not in CANDIDATE_SESSIONS:
        CANDIDATE_SESSIONS[app_no] = {
            "name": cand["name"], "logged_in_at": now_iso(),
            "started": False, "submitted": False,
            "start_time": None, "end_time": None, "answers": {},
        }
        log_event(app_no, "Candidate Login", 100)
    return ok({"candidate": {"app_number": app_no, "name": cand["name"]},
               "exam": EXAM_CONFIG}, "Login successful")

@app.route("/api/logout", methods=["GET","POST"])
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
    return ok({"app_number": app_no, "name": cand.get("name","Unknown"),
               "exam_started": sess.get("started",False),
               "exam_submitted": sess.get("submitted",False),
               "time_remaining": time_left(app_no)})


# ── Exam routes ───────────────────────────────────────────────

@app.route("/api/exam/info", methods=["GET"])
def exam_info():
    return ok({"exam": EXAM_CONFIG})

@app.route("/api/exam/start", methods=["POST"])
def exam_start():
    if "app_number" not in session: return err("Not authenticated.", 401)
    app_no = session["app_number"]
    d      = request.get_json(force=True) or {}
    if not d.get("agreed"): return err("You must agree to the instructions.")
    sess = CANDIDATE_SESSIONS.get(app_no, {})
    if sess.get("started"):
        return ok({"already_started": True, "time_remaining": time_left(app_no)}, "Exam already in progress")
    ts = time.time()
    CANDIDATE_SESSIONS[app_no].update({
        "started": True, "start_time": ts,
        "end_time": ts + EXAM_CONFIG["duration_mins"] * 60,
    })
    log_event(app_no, "Exam Started", 100)
    return ok({"start_time": ts, "end_time": CANDIDATE_SESSIONS[app_no]["end_time"],
               "duration_mins": EXAM_CONFIG["duration_mins"]}, "Exam started")

@app.route("/api/exam/answer", methods=["POST"])
def save_answer():
    if "app_number" not in session: return err("Not authenticated.", 401)
    app_no = session["app_number"]
    sess   = CANDIDATE_SESSIONS.get(app_no, {})
    if not sess.get("started"):        return err("Exam not started.")
    if sess.get("submitted"):          return err("Already submitted.")
    if time.time() > sess["end_time"]: return err("Time is up.")
    d   = request.get_json(force=True) or {}
    qid = d.get("question_id")
    if not qid: return err("question_id required.")
    sess["answers"][qid] = d.get("answer")
    return ok({"saved": qid})

@app.route("/api/exam/submit", methods=["POST"])
def exam_submit():
    if "app_number" not in session: return err("Not authenticated.", 401)
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
    if "app_number" not in session: return err("Not authenticated.", 401)
    app_no = session["app_number"]
    sess   = CANDIDATE_SESSIONS.get(app_no, {})
    return ok({"started": sess.get("started",False), "submitted": sess.get("submitted",False),
               "time_remaining": time_left(app_no),
               "answered_count": len(sess.get("answers",{}))})


# ── Teacher auth ──────────────────────────────────────────────

@app.route("/api/teacher/login", methods=["POST"])
def teacher_login():
    d    = request.get_json(force=True) or {}
    user = str(d.get("username","")).strip()
    pw   = str(d.get("password","")).strip()
    if not user or not pw: return err("Username and password required.")
    stored = TEACHERS.get(user)
    if not stored or stored != h(pw): return err("Invalid credentials.", 401)
    session["teacher"] = user
    session["role"]    = "teacher"
    return ok({"username": user}, "Teacher login successful")

@app.route("/api/teacher/logout", methods=["POST"])
def teacher_logout():
    session.clear()
    return ok(msg="Logged out")


# ── Teacher: student list ─────────────────────────────────────

@app.route("/api/teacher/students", methods=["GET"])
def teacher_students():
    students = []
    for app_no, sess in CANDIDATE_SESSIONS.items():
        evts = events_for(app_no)
        risk = risk_score_for(app_no)
        last = evts[0]["event_type"] if evts else "Logged In"
        if sess.get("submitted"):  label = "Submitted"
        elif sess.get("started"):  label = last
        else:                      label = "Waiting"
        students.append({
            "app_number": app_no, "name": sess["name"],
            "started": sess.get("started",False), "submitted": sess.get("submitted",False),
            "risk_score": risk, "event_count": len(evts), "last_event": label,
            "time_remaining": time_left(app_no),
            "answered_count": len(sess.get("answers",{})),
        })
    students.sort(key=lambda s: s["risk_score"], reverse=True)
    return ok({"students": students, "total": len(students)})


# ── Proctoring: events ────────────────────────────────────────

@app.route("/api/proctor/event", methods=["POST"])
def proctor_log():
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
    sf   = request.args.get("student","").lower()
    tf   = request.args.get("type","")
    evts = [e for e in PROCTOR_EVENTS
            if (not sf or sf in e["student_name"].lower())
            and (not tf or e["event_type"] == tf)]
    return ok({"events": evts, "total": len(evts)})

@app.route("/api/proctor/summary", methods=["GET"])
def proctor_summary():
    total  = len(CANDIDATE_SESSIONS)
    active = sum(1 for s in CANDIDATE_SESSIONS.values() if s.get("started") and not s.get("submitted"))
    submitted = sum(1 for s in CANDIDATE_SESSIONS.values() if s.get("submitted"))
    risk_map = {}
    for e in PROCTOR_EVENTS:
        if e["level"] == "info": continue
        risk_map[e["app_number"]] = min(100, risk_map.get(e["app_number"],0) + e["impact"])
    return ok({
        "logged_in": total, "active": active,
        "waiting": total - active - submitted, "submitted": submitted,
        "flagged":  sum(1 for v in risk_map.values() if v >= 70),
        "warnings": sum(1 for v in risk_map.values() if 30 <= v < 70),
        "total_events": sum(1 for e in PROCTOR_EVENTS if e["level"] != "info"),
    })

@app.route("/api/proctor/export", methods=["GET"])
def proctor_export():
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["Time","App Number","Student","Event","Confidence","Impact","Level","Note"])
    for e in PROCTOR_EVENTS:
        if e["level"] == "info": continue
        w.writerow([e["time"],e["app_number"],e["student_name"],e["event_type"],
                    str(e["confidence"])+"%","+"+str(e["impact"]),e["level"],e.get("note","")])
    out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=audit-report.csv"})

@app.route("/api/proctor/snapshot", methods=["POST"])
def proctor_snapshot():
    d = request.get_json(force=True) or {}
    image = d.get("image","")
    if not image or not image.startswith("data:image"):
        return err("Invalid image data.")
    if len(SNAPSHOTS) >= 500:
        SNAPSHOTS.pop()
    SNAPSHOTS.insert(0, {
        "id":           str(uuid.uuid4()),
        "time":         now_str(),
        "timestamp":    now_iso(),
        "app_number":   d.get("app_number",""),
        "student_name": d.get("student_name",""),
        "event_type":   d.get("event_type",""),
        "image":        image,
    })
    return ok(msg="Snapshot saved")

@app.route("/api/proctor/snapshots", methods=["GET"])
def proctor_snapshots():
    sf = request.args.get("student","").lower()
    tf = request.args.get("type","")
    snaps = [s for s in SNAPSHOTS
             if (not sf or sf in s["student_name"].lower())
             and (not tf or s["event_type"] == tf)]
    return ok({"snapshots": snaps, "total": len(snaps)})


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*52}\n  Open → http://localhost:{port}\n{'='*52}")
    print("  CANDIDATE: 240110012345 / Pass@1234")
    print("  TEACHER:   teacher1 / Teacher@123")
    print(f"{'='*52}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
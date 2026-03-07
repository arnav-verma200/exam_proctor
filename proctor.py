"""
=============================================================
  proctor.py  —  Standalone AI Vision Proctoring Engine

  Run this as a SEPARATE process from app.py when you want
  the vision engine on its own service (e.g. local dev with
  a real webcam instead of browser-pushed frames).

  In production, the merged app.py already includes the same
  vision logic — you do NOT need to run proctor.py separately.

  Install deps:
      pip install flask flask-cors opencv-python mediapipe requests numpy

  Run:
      python proctor.py

  Runs on http://localhost:5001

  Endpoints:
    GET  /video_feed      → MJPEG stream (teacher dashboard)
    POST /set_candidate   → { app_number, student_name }
    POST /clear_candidate → clear active candidate
    GET  /status          → current detection state JSON
    GET  /ping            → health check

  Detection cadence: ~20 fps
    • Face Not Detected  — 0 faces visible
    • Multiple Faces     — 2+ faces visible
    • Phone Detected     — large rectangular object (heuristic)

  On each flag:
    1. Cooldown check (per event type)
    2. POST event to Flask main app (/api/proctor/event)
    3. Buffer last 5 s of frames → save AVI clip → upload to Appwrite
=============================================================
"""

import base64
import io
import os
import threading
import time
import uuid
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

# URL of the main Flask exam portal (receives event POSTs)
FLASK_API = os.environ.get("FLASK_API", "http://localhost:5000")

# Appwrite cloud storage credentials for clip uploads
APPWRITE = {
    "endpoint":   "https://fra.cloud.appwrite.io/v1",
    "project_id": "69a1413700343030d7f6",
    "bucket_id":  "69a19b3d000c21c9bd5f",
}

# Local directory for temporary clip files before upload
CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

# Seconds to wait before re-firing the same event type (avoids log spam)
COOLDOWN: dict[str, int] = {
    "Face Not Detected": 8,
    "Multiple Faces":    8,
    "Phone Detected":    10,
}

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE INITIALISATION
# ─────────────────────────────────────────────────────────────

_mp_face  = mp.solutions.face_detection
_mp_hands = mp.solutions.hands
_mp_draw  = mp.solutions.drawing_utils

# full-range face model (model_selection=1) works up to ~5 m from camera
face_detector = _mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)

# Hand detector — used to support phone-near-hand heuristic
hands_detector = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────────────────────────────
#  SHARED STATE
#  All threads read/write this dict; writes are protected by state_lock.
# ─────────────────────────────────────────────────────────────

state: dict = {
    "app_number":     None,           # currently monitored candidate
    "student_name":   "Unknown",
    "face_count":     0,
    "phone_detected": False,
    "status_text":    "Waiting for candidate…",
    "status_color":   (180, 180, 180),  # BGR colour for overlay text
    "frame":          None,             # latest annotated JPEG bytes for MJPEG
    "last_flag":      {},               # event_type → last_fired timestamp
}
state_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
#  CAMERA INITIALISATION
# ─────────────────────────────────────────────────────────────

cap: cv2.VideoCapture | None = None
cap_lock = threading.Lock()


def _open_camera() -> None:
    """Try camera indices 0–3; store the first working one in `cap`."""
    global cap
    for idx in range(4):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_FPS,          30)
            print(f"[proctor] Camera opened at index {idx}")
            cap = c
            return
    print("[proctor] WARNING: No camera found — using blank frames.")


_open_camera()

# ─────────────────────────────────────────────────────────────
#  CLIP RECORDER  — rolling 5-second buffer
# ─────────────────────────────────────────────────────────────

class ClipRecorder:
    """
    Keeps a rolling 5-second buffer of raw BGR frames.

    Call trigger(event_type) to mark the current buffer for saving.
    Every 5 seconds the internal cycle thread checks for a pending
    trigger, saves an AVI clip, uploads it to Appwrite, and resets.
    """

    def __init__(self, fps: int = 20) -> None:
        self.fps = fps
        self._frames: list[np.ndarray] = []
        self._pending_event: str | None = None
        self._lock = threading.Lock()
        # Start the save-cycle in a daemon thread
        threading.Thread(target=self._cycle, daemon=True).start()

    def push(self, frame: np.ndarray) -> None:
        """Add a frame to the rolling buffer (trim to last 5 s)."""
        with self._lock:
            self._frames.append(frame.copy())
            max_frames = self.fps * 5
            if len(self._frames) > max_frames:
                self._frames.pop(0)

    def trigger(self, event_type: str) -> None:
        """Mark the current buffer for saving (first trigger wins)."""
        with self._lock:
            if self._pending_event is None:
                self._pending_event = event_type

    def _cycle(self) -> None:
        """Every 5 s: if a trigger is pending, save + upload the clip."""
        while True:
            time.sleep(5)
            with self._lock:
                event = self._pending_event
                frames = self._frames.copy()
                self._pending_event = None
                self._frames = []

            if event and frames:
                threading.Thread(
                    target=_save_and_upload_clip,
                    args=(frames, event, self.fps),
                    daemon=True,
                ).start()


clip_recorder = ClipRecorder(fps=20)

# ─────────────────────────────────────────────────────────────
#  CLIP HELPERS — save, upload, notify
# ─────────────────────────────────────────────────────────────

def _save_and_upload_clip(frames: list[np.ndarray], event_type: str, fps: int) -> None:
    """
    Write frames to a temporary AVI file, upload to Appwrite, then notify
    the Flask API. The local file is deleted after a successful upload.
    """
    app_no = state["app_number"]
    student_name = state["student_name"]
    if not app_no:
        return  # no active candidate — discard

    ts = int(time.time() * 1000)
    filename = f"{student_name}_{event_type.replace(' ', '_')}_{ts}.avi"
    filepath = os.path.join(CLIPS_DIR, filename)

    # Write AVI
    try:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"[proctor] Clip saved: {filepath}")
    except Exception as exc:
        print(f"[proctor] Clip write error: {exc}")
        return

    # Upload to Appwrite Storage
    file_id = _upload_to_appwrite(filepath, filename)
    if file_id:
        _notify_flask_clip(app_no, student_name, event_type, file_id, filename)

    # Remove local temp file regardless of upload success
    try:
        os.remove(filepath)
    except OSError:
        pass


def _upload_to_appwrite(filepath: str, filename: str) -> str | None:
    """
    POST the clip file to Appwrite Storage.

    Returns the Appwrite file ID on success, or None on failure.
    """
    file_id = ("clip" + str(uuid.uuid4()).replace("-", ""))[:36]
    url = (
        f"{APPWRITE['endpoint']}/storage/buckets/"
        f"{APPWRITE['bucket_id']}/files"
    )
    headers = {"X-Appwrite-Project": APPWRITE["project_id"]}

    try:
        with open(filepath, "rb") as fh:
            resp = requests.post(
                url,
                headers=headers,
                files={"file": (filename, fh, "video/x-msvideo")},
                data={"fileId": file_id},
                timeout=30,
            )
        if resp.status_code in (200, 201):
            returned_id = resp.json().get("$id", file_id)
            print(f"[proctor] Appwrite upload OK → {returned_id}")
            return returned_id
        else:
            print(f"[proctor] Appwrite upload failed: {resp.status_code} {resp.text[:200]}")
    except Exception as exc:
        print(f"[proctor] Appwrite upload error: {exc}")

    return None


def _notify_flask_clip(
    app_no: str,
    student_name: str,
    event_type: str,
    file_id: str,
    filename: str,
) -> None:
    """Inform the main Flask app that a clip has been uploaded."""
    try:
        requests.post(
            f"{FLASK_API}/api/proctor/clip",
            json={
                "app_number":   app_no,
                "student_name": student_name,
                "event_type":   event_type,
                "file_id":      file_id,
                "file_name":    filename,
                "bucket_id":    APPWRITE["bucket_id"],
            },
            timeout=5,
        )
    except Exception as exc:
        print(f"[proctor] Flask clip-notify error: {exc}")


def _notify_flask_event(event_type: str, confidence: int) -> None:
    """
    POST a proctoring event to the main Flask backend, respecting cooldowns.

    Also triggers clip recording so a clip is saved for each unique flag.
    """
    app_no = state["app_number"]
    if not app_no:
        return

    now = time.time()
    last_fired = state["last_flag"].get(event_type, 0)
    if now - last_fired < COOLDOWN.get(event_type, 5):
        return  # still in cooldown — skip

    state["last_flag"][event_type] = now

    try:
        requests.post(
            f"{FLASK_API}/api/proctor/event",
            json={
                "app_number": app_no,
                "event_type": event_type,
                "confidence": confidence,
            },
            timeout=3,
        )
        print(f"[proctor] Flagged → {event_type} ({confidence}%)")
        # Trigger a clip for this event
        clip_recorder.trigger(event_type)
    except Exception as exc:
        print(f"[proctor] Event notify error: {exc}")


# ─────────────────────────────────────────────────────────────
#  PHONE DETECTION HELPER
# ─────────────────────────────────────────────────────────────

def _detect_phone(
    frame_bgr: np.ndarray,
    face_bboxes: list[tuple],
) -> tuple[bool, int]:
    """
    Heuristic phone detection using contour analysis.

    Strategy:
      1. Greyscale + Gaussian blur + Canny edges + dilation
      2. Find external contours
      3. Accept contours whose:
           - Area is 3 %–35 % of the frame
           - Minimum bounding rect has aspect ratio 1.5–2.5 (portrait phone)
           - Centre does NOT overlap any detected face bbox
      4. Confidence scales with relative area

    Returns (detected: bool, confidence: int 0–100).
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

        aspect = max(rw, rh) / min(rw, rh)
        if not (1.5 <= aspect <= 2.5):
            continue

        cx, cy = int(rect[0][0]), int(rect[0][1])
        # Reject if the centre of the rectangle is inside a face bounding box
        overlaps_face = any(
            fx <= cx <= fx + fw and fy <= cy <= fy + fh
            for fx, fy, fw, fh in face_bboxes
        )
        if not overlaps_face:
            confidence = min(95, int(60 + (area / (w * h)) * 200))
            return True, confidence

    return False, 0


# ─────────────────────────────────────────────────────────────
#  MAIN DETECTION LOOP  (runs in a daemon thread)
# ─────────────────────────────────────────────────────────────

def _detection_loop() -> None:
    """
    Continuously:
      1. Read frame from webcam (or produce a blank frame if no camera)
      2. Run MediaPipe face detection
      3. Run phone heuristic
      4. Annotate the frame with overlays
      5. Push raw frame to clip recorder buffer
      6. Encode to JPEG → store in state["frame"] for MJPEG streaming
      7. Fire events to Flask if violations are detected
    """
    # Fallback blank frame shown when no camera is available
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        blank, "No Camera Found", (160, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 200), 2,
    )

    while True:
        # ── Grab frame ──
        frame: np.ndarray | None = None
        if cap and cap.isOpened():
            with cap_lock:
                ret, frame = cap.read()
            if not ret or frame is None:
                frame = blank.copy()
        else:
            frame = blank.copy()

        # Mirror (selfie-style) and convert for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # ── Face detection ──
        face_results = face_detector.process(rgb)
        face_count = 0
        face_bboxes: list[tuple] = []

        if face_results.detections:
            face_count = len(face_results.detections)
            for det in face_results.detections:
                bb = det.location_data.relative_bounding_box
                x  = max(0, int(bb.xmin * w))
                y  = max(0, int(bb.ymin * h))
                bw = int(bb.width  * w)
                bh = int(bb.height * h)
                face_bboxes.append((x, y, bw, bh))

                # Green box for a single face; blue for multiple
                box_color = (0, 220, 80) if face_count == 1 else (0, 80, 220)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
                score_label = f"Face {int(det.score[0] * 100)}%"
                cv2.putText(frame, score_label, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # ── Phone detection ──
        phone_detected, phone_conf = _detect_phone(frame, face_bboxes)

        # ── Determine status and fire violation events ──
        app_no = state["app_number"]

        if face_count == 0:
            status_txt   = "NO FACE DETECTED"
            status_color = (0, 0, 220)
            state["status_text"]  = "No Face"
            state["status_color"] = status_color
            if app_no:
                threading.Thread(
                    target=_notify_flask_event,
                    args=("Face Not Detected", 91),
                    daemon=True,
                ).start()

        elif face_count > 1:
            status_txt   = f"MULTIPLE FACES: {face_count}"
            status_color = (0, 160, 255)
            state["status_text"]  = f"Multiple Faces ({face_count})"
            state["status_color"] = status_color
            if app_no:
                threading.Thread(
                    target=_notify_flask_event,
                    args=("Multiple Faces", min(99, 80 + face_count * 5)),
                    daemon=True,
                ).start()

        else:
            status_txt   = "Face OK"
            status_color = (0, 200, 80)
            state["status_text"]  = "Face OK"
            state["status_color"] = status_color

        if phone_detected:
            state["phone_detected"] = True
            cv2.putText(
                frame,
                f"PHONE DETECTED ({phone_conf}%)",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 50, 200),
                2,
            )
            if app_no:
                threading.Thread(
                    target=_notify_flask_event,
                    args=("Phone Detected", phone_conf),
                    daemon=True,
                ).start()
        else:
            state["phone_detected"] = False

        state["face_count"] = face_count

        # ── Draw annotation overlay ──
        # Dark status bar at top
        cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
        cv2.putText(frame, status_txt, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # Candidate name top-right
        name = state["student_name"]
        if name and name != "Unknown":
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, name, (w - tw - 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Red border when any violation is active
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

        # Push raw frame into the rolling clip buffer
        clip_recorder.push(frame)

        # Encode to JPEG and store for the MJPEG stream
        ret2, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret2:
            with state_lock:
                state["frame"] = jpeg.tobytes()

        time.sleep(0.05)  # ~20 fps


# Start detection in a daemon thread (dies with the process)
threading.Thread(target=_detection_loop, daemon=True).start()

# ─────────────────────────────────────────────────────────────
#  FLASK APP  (port 5001 — separate from the main exam portal)
# ─────────────────────────────────────────────────────────────

vision_app = Flask(__name__)

_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5000,http://127.0.0.1:5000",
)
CORS(
    vision_app,
    supports_credentials=True,
    origins=[o.strip() for o in _raw_origins.split(",") if o.strip()],
)


def _generate_mjpeg():
    """Yield JPEG frames as a multipart MJPEG stream."""
    while True:
        with state_lock:
            frame = state.get("frame")
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(0.05)


@vision_app.route("/video_feed")
def video_feed():
    """MJPEG stream — embedded in the teacher dashboard as <img src>."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@vision_app.route("/set_candidate", methods=["POST"])
def set_candidate():
    """
    Set the currently monitored candidate.
    Called by the browser when a student starts the exam.
    Body: { "app_number": str, "student_name": str }
    """
    d = request.get_json(force=True) or {}
    state["app_number"]   = d.get("app_number")
    state["student_name"] = d.get("student_name", "Unknown")
    state["last_flag"]    = {}  # reset cooldowns for the new session
    print(f"[proctor] Candidate set → {state['student_name']} ({state['app_number']})")
    return jsonify({"status": "ok"})


@vision_app.route("/clear_candidate", methods=["POST"])
def clear_candidate():
    """
    Remove the active candidate (called when the exam window is closed).
    Resets vision state so the stream shows the idle placeholder.
    """
    state["app_number"]   = None
    state["student_name"] = "Unknown"
    return jsonify({"status": "ok"})


@vision_app.route("/status")
def status():
    """Return current detection state as JSON for the teacher dashboard widget."""
    return jsonify({
        "app_number":     state["app_number"],
        "student_name":   state["student_name"],
        "face_count":     state["face_count"],
        "phone_detected": state["phone_detected"],
        "status_text":    state["status_text"],
    })


@vision_app.route("/ping")
def ping():
    """Simple health-check endpoint."""
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    proctor_port = int(os.environ.get("PROCTOR_PORT", 5001))
    print(f"\n{'='*52}")
    print(f"  🔍  Proctor Vision Engine (standalone)")
    print(f"  Stream → http://localhost:{proctor_port}/video_feed")
    print(f"  Status → http://localhost:{proctor_port}/status")
    print(f"{'='*52}")
    print("  Detection: Face / Multiple Faces / Phone")
    print("  Clips saved to ./clips/ then uploaded to Appwrite")
    print(f"{'='*52}\n")
    vision_app.run(host="0.0.0.0", port=proctor_port, threaded=True)
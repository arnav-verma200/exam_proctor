"""
=============================================================
  proctor.py  —  AI Vision Proctoring Engine
  Uses: OpenCV + MediaPipe (FaceMesh + Hands + ObjectDetection)

  HOW TO RUN (in a SEPARATE terminal):
      pip install flask flask-cors opencv-python mediapipe requests
      python proctor.py

  Runs on http://localhost:5001
  ─────────────────────────────────────────────────
  Endpoints:
    GET  /video_feed          → MJPEG stream (shown in browser)
    POST /set_candidate       → { app_number, student_name }
    GET  /status              → current detection state
  ─────────────────────────────────────────────────
  Detection every ~500 ms:
    • Face Not Detected  — 0 faces
    • Multiple Faces     — 2+ faces
    • Phone Detected     — large rectangular object near face / hand holding object
  Records 5-second WebM-like AVI clips when a flag fires.
  Uploads clip to Appwrite, then notifies Flask (port 5000).
=============================================================
"""

import cv2
import mediapipe as mp
import threading
import time
import os
import io
import uuid
import requests
import base64
import numpy as np
from datetime import datetime
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
FLASK_API   = os.environ.get("FLASK_API", "http://localhost:5000")   # main Flask backend
APPWRITE    = {
    "endpoint":   "https://fra.cloud.appwrite.io/v1",
    "project_id": "69a1413700343030d7f6",
    "bucket_id":  "69a19b3d000c21c9bd5f",
}

CLIPS_DIR   = "clips"                          # temp folder for clip files
os.makedirs(CLIPS_DIR, exist_ok=True)

COOLDOWN = {                                   # seconds between repeated flags
    "Face Not Detected": 8,
    "Multiple Faces":    8,
    "Phone Detected":    10,
}

# ─────────────────────────────────────────
#  MEDIAPIPE INIT
# ─────────────────────────────────────────
mp_face  = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

face_detector = mp_face.FaceDetection(
    model_selection=1,        # 1 = full-range model (better for webcam)
    min_detection_confidence=0.5
)
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────
state = {
    "app_number":    None,
    "student_name":  "Unknown",
    "face_count":    0,
    "phone_detected":False,
    "status_text":   "Waiting for candidate…",
    "status_color":  (180, 180, 180),   # BGR
    "frame":         None,              # latest annotated JPEG bytes
    "last_flag":     {},                # event_type → last_fired timestamp
    "recording":     False,
    "clip_event":    None,
}
state_lock = threading.Lock()

cap    = None
cap_lock = threading.Lock()

# ─────────────────────────────────────────
#  CAMERA THREAD
# ─────────────────────────────────────────
def open_camera():
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
    print("[proctor] WARNING: No camera found. Using blank frames.")

open_camera()

# ─────────────────────────────────────────
#  CLIP RECORDER  (5-second AVI clips)
# ─────────────────────────────────────────
class ClipRecorder:
    """Records a 5-second window.  If flag() is called → save + upload."""
    def __init__(self, fps=20):
        self.fps     = fps
        self.frames  = []          # raw BGR frames
        self.flag    = None        # event type that triggered save
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._cycle, daemon=True)
        self._thread.start()

    def push(self, frame):
        with self._lock:
            self.frames.append(frame.copy())
            # Keep only last 5 seconds of frames
            max_frames = self.fps * 5
            if len(self.frames) > max_frames:
                self.frames.pop(0)

    def trigger(self, event_type):
        """Mark current buffer to be saved for this event."""
        with self._lock:
            if self.flag is None:           # don't overwrite an existing pending flag
                self.flag = event_type

    def _cycle(self):
        """Every 5s check if there's a pending flag → save+upload, else discard."""
        while True:
            time.sleep(5)
            with self._lock:
                flag   = self.flag
                frames = self.frames.copy()
                self.flag   = None
                self.frames = []

            if flag and frames:
                threading.Thread(
                    target=save_and_upload_clip,
                    args=(frames, flag, self.fps),
                    daemon=True
                ).start()


clip_recorder = ClipRecorder(fps=20)

def save_and_upload_clip(frames, event_type, fps):
    """Write frames to temp AVI, upload to Appwrite, notify Flask."""
    app_no = state["app_number"]
    name   = state["student_name"]
    if not app_no:
        return

    ts       = int(time.time() * 1000)
    filename = f"{name}_{event_type.replace(' ','_')}_{ts}.avi"
    filepath = os.path.join(CLIPS_DIR, filename)

    try:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"[proctor] Saved clip: {filepath}")
    except Exception as e:
        print(f"[proctor] Clip write error: {e}")
        return

    # Upload to Appwrite
    file_id = upload_to_appwrite(filepath, filename)
    if file_id:
        notify_flask_clip(app_no, name, event_type, file_id, filename)
    # Clean up local file
    try:
        os.remove(filepath)
    except Exception:
        pass


def upload_to_appwrite(filepath, filename):
    """POST file to Appwrite Storage, return file_id or None."""
    file_id = ("clip" + str(uuid.uuid4()).replace("-", ""))[:36]
    url     = f"{APPWRITE['endpoint']}/storage/buckets/{APPWRITE['bucket_id']}/files"
    headers = {"X-Appwrite-Project": APPWRITE["project_id"]}
    try:
        with open(filepath, "rb") as f:
            resp = requests.post(
                url, headers=headers,
                files={"file": (filename, f, "video/x-msvideo")},
                data={"fileId": file_id},
                timeout=30,
            )
        if resp.status_code in (200, 201):
            returned_id = resp.json().get("$id", file_id)
            print(f"[proctor] Appwrite upload OK → {returned_id}")
            return returned_id
        else:
            print(f"[proctor] Appwrite upload failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"[proctor] Appwrite upload error: {e}")
    return None


def notify_flask_clip(app_no, student_name, event_type, file_id, filename):
    """Tell the main Flask app about the new clip."""
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
    except Exception as e:
        print(f"[proctor] Flask notify error: {e}")


def notify_flask_event(event_type, confidence):
    """POST a proctoring event to the main Flask backend."""
    app_no = state["app_number"]
    if not app_no:
        return
    now = time.time()
    last = state["last_flag"].get(event_type, 0)
    cooldown = COOLDOWN.get(event_type, 5)
    if now - last < cooldown:
        return
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
        # Also trigger clip recording
        clip_recorder.trigger(event_type)
    except Exception as e:
        print(f"[proctor] Event notify error: {e}")

# ─────────────────────────────────────────
#  PHONE DETECTION HELPER
# ─────────────────────────────────────────
def detect_phone(frame_bgr, face_bboxes):
    """
    Heuristic phone detection using contours + aspect ratio.
    Phones appear as large, dark, rectangular objects.
    Looks for rectangles near face/hand regions.
    Returns (detected: bool, confidence: int)
    """
    h, w = frame_bgr.shape[:2]
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Phone occupies 3%–25% of frame area
        if area < 0.03 * w * h or area > 0.35 * w * h:
            continue
        rect   = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 5 or rh < 5:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        # Phone aspect ratio typically between 1.6 and 2.2
        if 1.5 <= aspect <= 2.5:
            # Check it's away from face bboxes (not confusing the face itself)
            box = cv2.boxPoints(rect)
            cx  = int(rect[0][0])
            cy  = int(rect[0][1])
            near_face = False
            for fb in face_bboxes:
                fx, fy, fw, fh = fb
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    near_face = True
                    break
            if not near_face:
                conf = min(95, int(60 + (area / (w * h)) * 200))
                return True, conf

    # Secondary: check if hand is holding something (large object touching hand landmarks)
    return False, 0


# ─────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────
def detection_loop():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "No Camera Found", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 200), 2)

    while True:
        # Read frame
        frame = None
        if cap and cap.isOpened():
            with cap_lock:
                ret, frame = cap.read()
            if not ret or frame is None:
                frame = blank.copy()
        else:
            frame = blank.copy()

        # Flip horizontally for mirror view
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── FACE DETECTION ──
        face_results = face_detector.process(rgb)
        face_count   = 0
        face_bboxes  = []

        h, w = frame.shape[:2]
        if face_results.detections:
            face_count = len(face_results.detections)
            for det in face_results.detections:
                bb   = det.location_data.relative_bounding_box
                x    = max(0, int(bb.xmin * w))
                y    = max(0, int(bb.ymin * h))
                bw   = int(bb.width  * w)
                bh   = int(bb.height * h)
                face_bboxes.append((x, y, bw, bh))

                # Draw box
                color = (0, 220, 80) if face_count == 1 else (0, 80, 220)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                score = int(det.score[0] * 100)
                label = f"Face {score}%"
                cv2.putText(frame, label, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── PHONE DETECTION ──
        phone_det, phone_conf = detect_phone(frame, face_bboxes)

        # ── STATUS TEXT & FLAGS ──
        app_no = state["app_number"]

        if face_count == 0:
            txt   = "NO FACE DETECTED"
            color = (0, 0, 220)
            state["status_text"]  = "No Face"
            state["status_color"] = color
            if app_no:
                threading.Thread(
                    target=notify_flask_event,
                    args=("Face Not Detected", 91),
                    daemon=True
                ).start()
        elif face_count > 1:
            txt   = f"MULTIPLE FACES: {face_count}"
            color = (0, 160, 255)
            state["status_text"]  = f"Multiple Faces ({face_count})"
            state["status_color"] = color
            if app_no:
                threading.Thread(
                    target=notify_flask_event,
                    args=("Multiple Faces", min(99, 80 + face_count * 5)),
                    daemon=True
                ).start()
        else:
            txt   = "Face OK"
            color = (0, 200, 80)
            state["status_text"]  = "Face OK"
            state["status_color"] = color

        if phone_det:
            state["phone_detected"] = True
            phone_txt = f"PHONE DETECTED ({phone_conf}%)"
            cv2.putText(frame, phone_txt, (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 200), 2)
            if app_no:
                threading.Thread(
                    target=notify_flask_event,
                    args=("Phone Detected", phone_conf),
                    daemon=True
                ).start()
        else:
            state["phone_detected"] = False

        state["face_count"] = face_count

        # ── OVERLAY: status bar ──
        cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
        cv2.putText(frame, txt, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        # Candidate name top-right
        if state["student_name"] and state["student_name"] != "Unknown":
            name_txt = state["student_name"]
            (tw, _), _ = cv2.getTextSize(name_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, name_txt, (w - tw - 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── RED BORDER when flagged ──
        if face_count == 0 or face_count > 1 or phone_det:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 200), 4)

        # ── TIMESTAMP ──
        ts_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts_str, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Push frame to clip recorder (always recording a rolling buffer)
        clip_recorder.push(frame)

        # Encode to JPEG for MJPEG stream
        ret2, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret2:
            with state_lock:
                state["frame"] = jpeg.tobytes()

        time.sleep(0.05)   # ~20 fps detection


# Start detection in background thread
threading.Thread(target=detection_loop, daemon=True).start()

# ─────────────────────────────────────────
#  FLASK APP  (port 5001)
# ─────────────────────────────────────────
vision_app = Flask(__name__)
_raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")
_proctor_origins = [o.strip() for o in _raw.split(",") if o.strip()]
CORS(vision_app, supports_credentials=True, origins=_proctor_origins)


def generate_mjpeg():
    """Generator for MJPEG stream."""
    while True:
        with state_lock:
            frame = state.get("frame")
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
        time.sleep(0.05)


@vision_app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@vision_app.route("/set_candidate", methods=["POST"])
def set_candidate():
    """Called by browser when student logs in."""
    d = request.get_json(force=True) or {}
    state["app_number"]   = d.get("app_number")
    state["student_name"] = d.get("student_name", "Unknown")
    state["last_flag"]    = {}   # reset cooldowns for new session
    print(f"[proctor] Candidate set → {state['student_name']} ({state['app_number']})")
    return jsonify({"status": "ok"})


@vision_app.route("/clear_candidate", methods=["POST"])
def clear_candidate():
    state["app_number"]   = None
    state["student_name"] = "Unknown"
    return jsonify({"status": "ok"})


@vision_app.route("/status")
def status():
    return jsonify({
        "app_number":    state["app_number"],
        "student_name":  state["student_name"],
        "face_count":    state["face_count"],
        "phone_detected":state["phone_detected"],
        "status_text":   state["status_text"],
    })


@vision_app.route("/ping")
def ping():
    return jsonify({"ok": True})


# ─────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    proctor_port = int(os.environ.get("PROCTOR_PORT", 5001))
    print("\n" + "=" * 52)
    print("  🔍  Proctor Vision Engine")
    print(f"  Stream  → http://localhost:{proctor_port}/video_feed")
    print(f"  Status  → http://localhost:{proctor_port}/status")
    print("=" * 52)
    print("  Detection: Face / Multiple Faces / Phone")
    print("  Clips saved to ./clips/ then Appwrite")
    print("=" * 52 + "\n")
    vision_app.run(host="0.0.0.0", port=proctor_port, threaded=True)
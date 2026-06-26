"""
Vision engine — MediaPipe face detection + heuristic phone detection.

Kept in a standalone module so Agent 2 can further optimise it
without touching route or service code.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from backend.services.event_service import log_event

_face_detector: mp.solutions.face_detection.FaceDetection | None = None
_face_detector_lock = threading.Lock()

VISION_COOLDOWN = {
    "Face Not Detected": 8,
    "Multiple Faces": 8,
    "Phone Detected": 10,
}
FRAME_TIMEOUT = 10

candidate_frames: dict[str, dict] = {}
frames_lock = threading.Lock()

vision_state: dict = {
    "app_number": None,
    "student_name": "Unknown",
    "face_count": 0,
    "phone_detected": False,
    "status_text": "Waiting for candidate…",
    "last_flag": {},
    "output_frame": None,
}
vision_lock = threading.Lock()

# ── ThreadPoolExecutor + frame processing pipeline ──
_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_frame_queue: queue.Queue = queue.Queue(maxsize=5)
frames_dropped = 0

# Output buffer — holds only the latest annotated frame for MJPEG
_output_deque: deque = deque(maxlen=1)
_output_deque_lock = threading.Lock()

# Serialises MediaPipe inference (not thread-safe across workers)
_inference_lock = threading.Lock()

_detection_thread_started = False
_detection_thread_lock = threading.Lock()


def _get_face_detector() -> mp.solutions.face_detection.FaceDetection:
    global _face_detector
    with _face_detector_lock:
        if _face_detector is None:
            _face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )
    return _face_detector


def _make_placeholder(text: str = "Waiting for candidate…") -> bytes:
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
        if not (1.5 <= max(rw, rh) / min(rw, rh) <= 2.5):
            continue
        cx, cy = int(rect[0][0]), int(rect[0][1])
        overlaps_face = any(
            fx <= cx <= fx + fw and fy <= cy <= fy + fh
            for fx, fy, fw, fh in face_bboxes
        )
        if not overlaps_face:
            confidence = min(95, int(60 + (area / (w * h)) * 200))
            return True, confidence
    return False, 0


def _notify_proctor_event(event_type: str, confidence: int) -> None:
    app_no = vision_state.get("app_number")
    if not app_no:
        return

    now = time.time()
    cooldown = VISION_COOLDOWN.get(event_type, 5)
    last_fired = vision_state.get("last_flag", {}).get(event_type, 0)
    if now - last_fired < cooldown:
        return

    with vision_lock:
        vision_state.setdefault("last_flag", {})[event_type] = now

    log_event(app_no, event_type, confidence)
    import logging
    logging.getLogger("vision").warning("%s (%s%%) → %s", event_type, confidence, app_no)


def store_frame(app_number: str, raw_jpeg: bytes) -> None:
    with frames_lock:
        candidate_frames[app_number] = {
            "raw_jpeg": raw_jpeg,
            "last_seen": time.time(),
        }
    with vision_lock:
        if vision_state.get("app_number") is None:
            vision_state["app_number"] = app_number


def set_active_candidate(app_number: str | None, student_name: str = "Unknown") -> None:
    with vision_lock:
        vision_state["app_number"] = app_number
        vision_state["student_name"] = student_name
        vision_state["last_flag"] = {}


def clear_active_candidate() -> None:
    with vision_lock:
        vision_state["app_number"] = None
        vision_state["student_name"] = "Unknown"


def _process_frame(raw_jpeg: bytes) -> None:
    """Run face + phone detection on a raw JPEG frame (executor worker).

    Decodes the JPEG, runs inference, annotates the frame, and stores the
    result in the output deque for MJPEG streaming.
    """
    try:
        arr = np.frombuffer(raw_jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # ── Face detection (serialised — MediaPipe is not thread-safe) ──
    with _inference_lock:
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
            box_color = (0, 220, 80) if face_count == 1 else (0, 80, 220)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
            cv2.putText(frame, f"Face {int(det.score[0] * 100)}%", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # ── Phone detection ──
    phone_detected, phone_conf = _detect_phone(frame, face_bboxes)

    # ── Update shared state atomically ──
    with vision_lock:
        if face_count == 0:
            vision_state["status_text"] = "No Face"
            status_txt = "NO FACE DETECTED"
            status_color = (0, 0, 220)
        elif face_count > 1:
            vision_state["status_text"] = f"Multiple Faces ({face_count})"
            status_txt = f"MULTIPLE FACES: {face_count}"
            status_color = (0, 160, 255)
        else:
            vision_state["status_text"] = "Face OK"
            status_txt = "Face OK"
            status_color = (0, 200, 80)
        vision_state["phone_detected"] = phone_detected
        vision_state["face_count"] = face_count
        student_name = vision_state.get("student_name", "Unknown")

    # ── Annotation overlay ──
    cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.putText(frame, status_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    if student_name and student_name != "Unknown":
        (tw, _), _ = cv2.getTextSize(student_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, student_name, (w - tw - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if face_count == 0 or face_count > 1 or phone_detected:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 200), 4)

    if phone_detected:
        cv2.putText(frame, f"PHONE ({phone_conf}%)", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 200), 2)

    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── Encode and store ──
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with _output_deque_lock:
        _output_deque.append(jpeg.tobytes())

    # ── Fire violation events (non-blocking daemon threads) ──
    if face_count == 0:
        threading.Thread(
            target=_notify_proctor_event,
            args=("Face Not Detected", 91),
            daemon=True,
        ).start()
    elif face_count > 1:
        threading.Thread(
            target=_notify_proctor_event,
            args=("Multiple Faces", min(99, 80 + face_count * 5)),
            daemon=True,
        ).start()
    if phone_detected:
        threading.Thread(
            target=_notify_proctor_event,
            args=("Phone Detected", phone_conf),
            daemon=True,
        ).start()


def _consumer_thread() -> None:
    """Pick up latest frame for the active candidate and submit to executor.

    Frame-drop logic: if the queue is full, the oldest pending frame is
    evicted so the newest frame is always processed.
    """
    global frames_dropped
    _last_submitted_ts: dict[str, float] = {}

    while True:
        app_no = vision_state.get("app_number")
        raw_jpeg: bytes | None = None
        ts: float = 0.0

        if app_no:
            with frames_lock:
                slot = candidate_frames.get(app_no)
                if slot and (time.time() - slot["last_seen"]) <= FRAME_TIMEOUT:
                    raw_jpeg = slot["raw_jpeg"]
                    ts = slot["last_seen"]

            if raw_jpeg and _last_submitted_ts.get(app_no) != ts:
                _last_submitted_ts[app_no] = ts
                try:
                    _frame_queue.put_nowait(raw_jpeg)
                except queue.Full:
                    frames_dropped += 1
                    # Evict oldest pending frame so newest is processed
                    try:
                        _frame_queue.get_nowait()
                        _frame_queue.put_nowait(raw_jpeg)
                    except queue.Empty:
                        pass
        else:
            # No active candidate — show placeholder
            with _output_deque_lock:
                _output_deque.append(_make_placeholder("No candidate active"))

        time.sleep(0.016)  # ~60 Hz poll rate


def get_latest_annotated_frame() -> bytes | None:
    """Return the most recently annotated JPEG, or None if none yet."""
    with _output_deque_lock:
        if _output_deque:
            return _output_deque[-1]
    return None


def ensure_detection_thread() -> None:
    global _detection_thread_started
    with _detection_thread_lock:
        if not _detection_thread_started:
            _detection_thread_started = True
            t = threading.Thread(target=_consumer_thread, daemon=True)
            t.start()

            def _frame_worker() -> None:
                """Pull frames from the queue and submit to the executor."""
                while True:
                    raw = _frame_queue.get()
                    _EXECUTOR.submit(_process_frame, raw)

            threading.Thread(target=_frame_worker, daemon=True).start()
            # Pre-warm executor threads by submitting a no-op
            _EXECUTOR.submit(lambda: None)

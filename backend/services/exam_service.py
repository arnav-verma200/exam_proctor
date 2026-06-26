import time
from datetime import datetime

EXAM_CONFIG: dict = {
    "title": "JEE (Main) 2026",
    "paper": "Paper 1 (B.E. / B.Tech)",
    "duration_mins": 180,
    "total_questions": 75,
    "total_marks": 300,
}

CANDIDATE_SESSIONS: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now().isoformat()


def get_session(app_number: str) -> dict:
    return CANDIDATE_SESSIONS.get(app_number, {})


def create_session(app_number: str, name: str) -> dict:
    if app_number not in CANDIDATE_SESSIONS:
        CANDIDATE_SESSIONS[app_number] = {
            "name": name,
            "logged_in_at": _now_iso(),
            "started": False,
            "submitted": False,
            "start_time": None,
            "end_time": None,
            "answers": {},
        }
    return CANDIDATE_SESSIONS[app_number]


def time_left(app_number: str) -> int:
    sess = CANDIDATE_SESSIONS.get(app_number, {})
    if not sess.get("started") or sess.get("submitted"):
        return 0
    return max(0, int(sess["end_time"] - time.time()))


def start_exam(app_number: str, duration_mins: int) -> dict:
    ts = time.time()
    CANDIDATE_SESSIONS[app_number].update({
        "started": True,
        "start_time": ts,
        "end_time": ts + duration_mins * 60,
    })
    return {
        "start_time": ts,
        "end_time": CANDIDATE_SESSIONS[app_number]["end_time"],
        "duration_mins": duration_mins,
    }


def save_answer(app_number: str, question_id: str, answer) -> bool:
    sess = CANDIDATE_SESSIONS.get(app_number, {})
    if not sess.get("started") or sess.get("submitted"):
        return False
    if time.time() > sess.get("end_time", 0):
        return False
    sess["answers"][question_id] = answer
    return True


def submit_exam(app_number: str, answers: dict | None = None) -> int | None:
    sess = CANDIDATE_SESSIONS.get(app_number)
    if not sess or not sess.get("started") or sess.get("submitted"):
        return None
    if answers:
        sess["answers"].update(answers)
    sess["submitted"] = True
    sess["submit_time"] = time.time()
    return len(sess["answers"])


def get_exam_status(app_number: str) -> dict:
    sess = CANDIDATE_SESSIONS.get(app_number, {})
    return {
        "started": sess.get("started", False),
        "submitted": sess.get("submitted", False),
        "time_remaining": time_left(app_number),
        "answered_count": len(sess.get("answers", {})),
    }

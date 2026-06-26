import csv
import io
import uuid
from datetime import datetime

from flask import Response, current_app

EVENT_META: dict[str, dict] = {
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

PROCTOR_EVENTS: list[dict] = []
SNAPSHOTS: list[dict] = []


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _now_iso() -> str:
    return datetime.now().isoformat()


def log_event(
    app_no: str,
    event_type: str,
    confidence: int,
    impact_override: int | None = None,
    note: str = "",
    candidate_name: str = "",
) -> dict:
    meta = EVENT_META.get(event_type, {"level": "danger", "impact": 10})
    impact = impact_override if impact_override is not None else meta["impact"]
    event = {
        "id": str(uuid.uuid4()),
        "time": _now_str(),
        "timestamp": _now_iso(),
        "app_number": app_no,
        "student_name": candidate_name or app_no,
        "event_type": event_type,
        "level": meta["level"],
        "confidence": confidence,
        "impact": impact,
        "note": note,
    }
    PROCTOR_EVENTS.insert(0, event)
    return event


def events_for(app_no: str) -> list[dict]:
    return [e for e in PROCTOR_EVENTS if e["app_number"] == app_no and e["level"] != "info"]


def filter_events(student_filter: str = "", type_filter: str = "") -> list[dict]:
    return [
        e for e in PROCTOR_EVENTS
        if (not student_filter or student_filter in e["student_name"].lower())
        and (not type_filter or e["event_type"] == type_filter)
    ]


def export_csv() -> Response:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Time", "App Number", "Student", "Event", "Confidence", "Impact", "Level", "Note"])
    for e in PROCTOR_EVENTS:
        if e["level"] == "info":
            continue
        writer.writerow([
            e["time"], e["app_number"], e["student_name"],
            e["event_type"], f"{e['confidence']}%", f"+{e['impact']}",
            e["level"], e.get("note", ""),
        ])
    out.seek(0)
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit-report.csv"},
    )


def add_snapshot(app_number: str, student_name: str, event_type: str, image: str) -> dict:
    if len(SNAPSHOTS) >= 500:
        SNAPSHOTS.pop()
    snap = {
        "id": str(uuid.uuid4()),
        "time": _now_str(),
        "timestamp": _now_iso(),
        "app_number": app_number,
        "student_name": student_name,
        "event_type": event_type,
        "image": image,
    }
    SNAPSHOTS.insert(0, snap)
    return snap


def filter_snapshots(student_filter: str = "", type_filter: str = "") -> list[dict]:
    return [
        s for s in SNAPSHOTS
        if (not student_filter or student_filter in s["student_name"].lower())
        and (not type_filter or s["event_type"] == type_filter)
    ]

import time



from backend.services.event_service import EVENT_META, PROCTOR_EVENTS, events_for

COOLDOWN_DEFAULTS = {
    "Face Not Detected": 8,
    "Multiple Faces": 8,
    "Phone Detected": 10,
}



def calculate_risk_from_events(app_number: str) -> int:
    return min(
        100,
        sum(e["impact"] for e in PROCTOR_EVENTS if e["app_number"] == app_number and e["level"] != "info"),
    )


def build_summary() -> dict:
    risk_map: dict[str, int] = {}
    for e in PROCTOR_EVENTS:
        if e["level"] == "info":
            continue
        risk_map[e["app_number"]] = min(100, risk_map.get(e["app_number"], 0) + e["impact"])

    from backend.services.exam_service import CANDIDATE_SESSIONS
    total = len(CANDIDATE_SESSIONS)
    active = sum(1 for s in CANDIDATE_SESSIONS.values() if s.get("started") and not s.get("submitted"))
    submitted = sum(1 for s in CANDIDATE_SESSIONS.values() if s.get("submitted"))

    return {
        "logged_in": total,
        "active": active,
        "waiting": total - active - submitted,
        "submitted": submitted,
        "flagged": sum(1 for v in risk_map.values() if v >= 70),
        "warnings": sum(1 for v in risk_map.values() if 30 <= v < 70),
        "total_events": sum(1 for e in PROCTOR_EVENTS if e["level"] != "info"),
    }

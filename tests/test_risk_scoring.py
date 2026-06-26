from backend.services.event_service import EVENT_META, events_for, log_event
from backend.services.proctor_service import calculate_risk_from_events


def test_risk_score_sums_impacts(seeded_events):
    score = calculate_risk_from_events("240110012345")
    expected = (
        EVENT_META["Tab Switch"]["impact"]
        + EVENT_META["Face Not Detected"]["impact"]
        + EVENT_META["Phone Detected"]["impact"]
    )
    assert score == expected


def test_risk_score_caps_at_100():
    log_event("240110012345", "Phone Detected", 95)
    log_event("240110012345", "Phone Detected", 95)
    log_event("240110012345", "Phone Detected", 95)
    score = calculate_risk_from_events("240110012345")
    assert score == 100


def test_info_events_do_not_affect_risk():
    log_event("240110012345", "Exam Started", 100)
    log_event("240110012345", "Exam Submitted", 100)
    log_event("240110012345", "Candidate Login", 100)
    score = calculate_risk_from_events("240110012345")
    assert score == 0


def test_risk_score_for_candidate_with_no_events():
    score = calculate_risk_from_events("240110099001")
    assert score == 0


def test_risk_score_mix_of_warn_and_danger():
    log_event("240110012345", "Tab Switch", 50)
    log_event("240110012345", "Browser Blur", 40)
    log_event("240110012345", "Multiple Faces", 90)
    score = calculate_risk_from_events("240110012345")
    expected = (
        EVENT_META["Tab Switch"]["impact"]
        + EVENT_META["Browser Blur"]["impact"]
        + EVENT_META["Multiple Faces"]["impact"]
    )
    assert score == expected


def test_event_type_default_impact():
    log_event("240110012345", "Unknown Event Type", 50)
    score = calculate_risk_from_events("240110012345")
    assert score == 10


def test_events_for_returns_non_info_only(seeded_events):
    evts = events_for("240110012345")
    for e in evts:
        assert e["level"] != "info"
    assert len(evts) == 3


def test_impact_override():
    log_event("240110012345", "Tab Switch", 50, impact_override=50)
    evts = events_for("240110012345")
    assert evts[0]["impact"] == 50

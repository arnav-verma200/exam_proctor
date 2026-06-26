from backend.services.proctor_service import COOLDOWN_DEFAULTS


def test_cooldown_periods_from_config():
    assert COOLDOWN_DEFAULTS["Face Not Detected"] == 8
    assert COOLDOWN_DEFAULTS["Multiple Faces"] == 8
    assert COOLDOWN_DEFAULTS["Phone Detected"] == 10


def test_unknown_event_default_cooldown():
    assert COOLDOWN_DEFAULTS.get("Unknown Event", 5) == 5


def test_cooldown_default_fallback():
    assert COOLDOWN_DEFAULTS.get("Tab Switch", 5) == 5

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["JWT_SECRET_KEY"] = "test-jwt-secret"
os.environ["SECRET_KEY"] = "test-flask-secret"
os.environ["FLASK_ENV"] = "testing"

test_engine = create_engine("sqlite:///:memory:", echo=False)

class Base(DeclarativeBase):
    pass

Base.metadata.create_all(test_engine)

cv2_mock = MagicMock()
cv2_mock.imencode.return_value = (True, MagicMock(tobytes=MagicMock(return_value=b"fake-jpeg")))
mp_mock = MagicMock()
mp_fd = MagicMock()
mp_fd.FaceDetection.return_value.process.return_value.detections = []
sys.modules["cv2"] = cv2_mock
sys.modules["mediapipe"] = mp_mock
sys.modules["mediapipe.solutions"] = mp_mock.solutions
sys.modules["mediapipe.solutions.face_detection"] = mp_fd
sys.modules["mediapipe.solutions.hands"] = MagicMock()
sys.modules["mediapipe.solutions.drawing_utils"] = MagicMock()

redis_mock = MagicMock()
redis_mock.from_url.return_value = redis_mock
redis_mock.exists.return_value = False
redis_mock.get.return_value = "0"
redis_mock.set.return_value = True
redis_mock.setex.return_value = True
redis_mock.pipeline.return_value = redis_mock
redis_mock.zremrangebyscore.return_value = redis_mock
redis_mock.zcard.return_value = 0
redis_mock.zadd.return_value = redis_mock
redis_mock.expire.return_value = redis_mock
redis_mock.execute.return_value = (0, 0, 1, True)
sys.modules["redis"] = redis_mock

db_session_mock = MagicMock()
db_session_mock.query.return_value.filter.return_value.first.return_value = None
db_session_mock.query.return_value.filter.return_value.count.return_value = 0
db_session_mock.query.return_value.filter.return_value.all.return_value = []
db_session_mock.query.return_value.count.return_value = 0
db_session_mock.query.return_value.join.return_value.filter.return_value.all.return_value = []
db_session_mock.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = []
db_session_mock.query.return_value.filter.return_value.with_entities.return_value.scalar.return_value = 0
db_session_mock.add = MagicMock()
db_session_mock.flush = MagicMock()
db_session_mock.commit = MagicMock()
db_session_mock.close = MagicMock()

db_mod = ModuleType("backend.database")
db_mod.Base = Base
db_mod.SessionLocal = MagicMock(return_value=db_session_mock)
db_mod.get_db = MagicMock(return_value=db_session_mock)
db_mod.init_db = MagicMock()
db_mod.engine = test_engine
db_mod.create_engine = create_engine
sys.modules["backend.database"] = db_mod
sys.modules["backend.database"].__dict__.update({
    "Base": Base, "SessionLocal": db_mod.SessionLocal,
    "get_db": db_mod.get_db, "init_db": db_mod.init_db,
    "engine": test_engine, "create_engine": create_engine,
})

from backend.services.event_service import EVENT_META, PROCTOR_EVENTS, SNAPSHOTS
from backend.services.exam_service import CANDIDATE_SESSIONS, EXAM_CONFIG
from backend.services.auth_service import CANDIDATES, TEACHERS
from backend.services.proctor_service import COOLDOWN_DEFAULTS


@pytest.fixture(autouse=True)
def reset_state():
    PROCTOR_EVENTS.clear()
    SNAPSHOTS.clear()
    CANDIDATE_SESSIONS.clear()


@pytest.fixture
def seeded_events():
    from backend.services.event_service import log_event
    log_event("240110012345", "Tab Switch", 80)
    log_event("240110012345", "Face Not Detected", 91)
    log_event("240110012345", "Phone Detected", 75)
    log_event("240110012345", "Exam Started", 100)
    log_event("240110056789", "Browser Blur", 30)
    yield


@pytest.fixture
def app():
    from backend import create_app

    flask_app = create_app()
    flask_app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "REDIS_CLIENT": redis_mock,
    })
    with flask_app.app_context():
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()

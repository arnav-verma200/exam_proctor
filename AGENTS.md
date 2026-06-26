# Exam Proctor — Agent Guide

## Entry points

- **Dev**: `python app.py` (port 5000) or `python main.py` (auto-opens browser)
- **Production**: `gunicorn 'backend:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- **Standalone proctor**: `python proctor.py` (port 5001, separate process)
- **Seed DB**: `python -m backend.seed` (safe to re-run)
- **Tests**: `pytest` (conftest mocks cv2, mediapipe, redis, and the DB module)

## Architecture

- **Factory**: `backend/__init__.py:create_app()` — blueprint registration, CORS, Redis init, DB init
- **Blueprints** (none use `url_prefix`; routes declare full paths):
  - `auth_bp` — `/api/login`, `/api/teacher/login`, `/api/session`, `/` (redirects to `/login_interface.html`)
  - `exam_bp` — `/api/exam/*` (start, answer, submit, status)
  - `proctor_bp` — `/proctor/*` (frames, video_feed, status) + `/api/proctor/*` (events, snapshots, summary)
  - `teacher_bp` — currently has only `/api/teacher/students`
- **Frontend**: static HTML/CSS/JS served from `frontend/` via `static_folder="../frontend"`, `static_url_path=""`
  - `connected_portal.html` (real app, ES modules) — served at `/login_interface.html` only
  - `login_interface.html` (legacy demo, no backend integration) — served at `/`
- **Auth**: hardcoded `CANDIDATES`/`TEACHERS` dicts in `backend/services/auth_service.py`. DB-backed `User` model exists but is not used for auth in current code.
- **Sessions**: in-memory `CANDIDATE_SESSIONS` dict in `exam_service.py` — lost on restart.

## Credentials (printed at startup)

- Candidate: `240110012345` / `Pass@1234`
- Teacher: `teacher1` / `Teacher@123`

## Key quirks & constraints

- **`--workers 1` is required** in production (detection state lives in process memory, not in Redis/DB). Procfile says `workers=2` — that's a bug.
- **`DATABASE_URL` env var is required** — raises `RuntimeError` if unset. Tests mock it to `sqlite:///:memory:`.
- **`JWT_SECRET_KEY` env var is required** — `jwt_service.py` raises `RuntimeError` if unset.
- **Alembic** manages DB migrations (`alembic.ini`). After schema changes run `alembic revision --autogenerate -m "msg"` then `alembic upgrade head`. Never use Drizzle (the previous AGENTS.md was wrong).
- **OpenCV + MediaPipe system deps** (Dockerfile): `libgl1-mesa-glx`, `libglib2.0-0`, `libsm6`, `libxext6`, `libgomp1`.
- **Two proctoring engines coexist**: browser-side (face-api.js, audio, tab monitoring in `proctoring.js`) and server-side (MediaPipe in `vision_service.py`). The server-side engine runs only when frames are pushed from the browser.
- **Redis** is used for cooldown tracking, rate limiting, and risk scores. Falls back gracefully if `REDIS_CLIENT` config is missing.
- **Rate limit**: `/proctor/frame/<app_number>` allows max 10 POSTs/second per candidate (Redis-backed).

## Frontend JS modules (`frontend/js/`)

- `api.js` — `apiRequest()` wrapper (JSON parse, error extraction from response)
- `auth.js` — `candidateLogin()`, `teacherLogin()`, `agreeAndStart()`, `setCandidate()`
- `proctoring.js` — camera init, frame pushing, face/audio/speech/tab monitoring
- `dashboard.js` — teacher dashboard: student list, charts, audit log, snapshots, polling (4s interval)

## Tests

- `conftest.py` mocks: cv2, mediapipe, redis, `backend.database` module (with SQLite in-memory engine)
- `reset_state` autouse fixture clears `PROCTOR_EVENTS`, `SNAPSHOTS`, `CANDIDATE_SESSIONS`
- Test helpers `_ok`/`_err` in each route module are duplicated (not imported from shared module)

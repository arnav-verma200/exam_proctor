# AI Proctored Exam Portal

Stack: Python Flask (backend) · Vanilla HTML/CSS/JS (frontend) · PostgreSQL · Redis · OpenCV + MediaPipe · Tailwind CSS v4

## Features

**Candidate:**

- Login with application number + password
- Timed exam with auto-submit on expiry
- Answer saving (per-question & bulk on submit)
- Browser-side proctoring: face-api.js detection, tab-switch & blur alerts, audio keyword spotting, camera snapshot capture on violations
- Pushes JPEG frames server-side (~5 fps)

**Teacher Dashboard:**

- Live MJPEG video stream from any candidate's webcam
- Per-student risk score (0–100) with color coding (green/amber/red)
- Real-time violation feed (polls every 4s) — Tab Switch, Face Not Detected, Multiple Faces, Phone Detected, Audio Detected, Browser Blur
- Snapshot evidence gallery with modal drill-down
- Summary stats: logged in, active, waiting, submitted, flagged, warnings
- CSV export of full audit trail

**Server-Side Proctoring:**

- MediaPipe face detection (single/multiple faces)
- Heuristic phone detection (contour analysis + aspect ratio)
- Per-event-type cooldowns (anti-spam)
- Risk score accumulator (impact-based, capped at 100)
- Rate-limited frame endpoint (10 req/s per candidate)
- ThreadPoolExecutor for frame analysis; frame-drop under load
- Structured JSON logging; Redis for cooldowns & rate limiting

## How to Run

### 1. Prerequisites

- Python 3.11+
- PostgreSQL (or use the Railway URL in .env)
- Redis (or use the Railway URL in .env)

### 2. Setup

```bash
git clone <repo>
cd exam_proctor
pip install -r requirements.txt
python -m backend.seed        # populate DB with users + exam
```

### 3. Environment

`.env` already has all secrets (Railway Redis, PostgreSQL, Appwrite, JWT). For local dev:

```bash
# Override DATABASE_URL & REDIS_URL with local instances if desired
```

### 4. Run (development)

```bash
python app.py
# → http://localhost:5000
```

### 5. Run (production — Railway)

```bash
gunicorn 'backend:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

Or deploy directly — Dockerfile is ready:

```bash
docker build -t exam-proctor .
docker run -p 5000:8080 --env-file .env exam-proctor
```

### 6. Test Credentials

| Role      | Username / App No | Password    |
| --------- | ----------------- | ----------- |
| Candidate | 240110012345      | Pass@1234   |
| Candidate | 240110056789      | Pass@5678   |
| Candidate | 240110099001      | Pass@9900   |
| Teacher   | teacher1          | Teacher@123 |
| Teacher   | admin             | Admin@2026  |

### 7. Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
backend/               # Flask app factory + blueprints
  auth/                #   candidate/teacher auth routes
  exam/                #   exam lifecycle routes
  proctor/             #   vision + event routes
  teacher/             #   teacher dashboard routes
  services/            #   business logic layer (auth, exam, event, proctor, vision, jwt)
  models/              #   Pydantic schemas + SQLAlchemy models
  database.py          #   PostgreSQL engine
  seed.py              #   idempotent DB seeder
frontend/              # Vanilla HTML + ES modules
  js/                  #   api.js, auth.js, dashboard.js, proctoring.js
  css/                 #   design system tokens
tests/                 # 30 passing tests (pytest)
Dockerfile             # Multi-stage build for Railway
Procfile               # Gunicorn entrypoint
```

## Key Environment Variables

```
DATABASE_URL      PostgreSQL connection string
REDIS_URL         Redis connection string
JWT_SECRET_KEY    HS256 signing key
APPWRITE_*        Appwrite storage for video clips
FLASK_ENV         production/development
```

## Known Conflicts (need human decision)

1. DB-backed services exist but aren't wired into routes (still uses in-memory)
2. Procfile says `--workers 2` but in-memory state requires `--workers 1`
3. JWT `require_auth` decorator exists but no routes use it yet

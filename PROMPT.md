PROJECT CONTEXT:
AI Proctored Exam Portal — online exam platform with real-time AI proctoring.

- Candidates: face detection, phone detection, audio keyword spotting, tab-switch alerts
- Teachers: live MJPEG feed, violation logs, risk scoring dashboard

TECH STACK:

- Backend: Python Flask (app.py), Gunicorn, Railway deployment
- Frontend: Vanilla HTML/CSS/JS (no framework)
- ML: face-api.js (browser), OpenCV-headless + MediaPipe (server-side)
- Storage: Appwrite (video clips)
- Skills installed: flask-python, redis-core, sqlalchemy-postgres, building-pydantic-ai-agents, docker-expert, pytest-coverage, tailwind-design-system

ENVIRONMENT:
All secrets are in .env at project root. Use python-dotenv to load them.
Variables: REDIS_URL, DATABASE_URL, APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID,
APPWRITE_API_KEY, APPWRITE_BUCKET_ID, JWT_SECRET_KEY

DESIGN SYSTEM:
Read @DESIGN.md at project root before touching any frontend file.
Every frontend decision must follow it exactly

---

YOU ARE THE ORCHESTRATOR. Spawn sub-agents for each domain below.
Do not do everything yourself. Delegate, then synthesize.

---

AGENT 1 — Backend Restructurer

- Split app.py into Flask blueprints: auth, exam, proctor, teacher
- Create a services/ layer — no business logic inside routes
- Use the flask-python skill for blueprint and app factory pattern
- Move all violation cooldown state and risk scoring out of memory into Redis
- Use redis-core skill for all Redis integration
- Add rate limiting on the frame-push endpoint
- Add input validation using Pydantic on all request schemas (use building-pydantic-ai-agents skill)
- Replace all print statements with structured JSON logging via Python logging module
- Ensure MJPEG streaming route is fully isolated so it never blocks other requests

AGENT 2 — ML / Proctoring Performance

- Move all OpenCV and MediaPipe inference off the request thread
- Implement ThreadPoolExecutor for frame analysis jobs
- Verify MJPEG generator uses proper multipart/x-mixed-replace headers
- Add frame drop logic under load so stream degrades gracefully instead of blocking
- Review violation event pipeline: cooldowns, thresholds, risk score calculation
- Fix any race conditions in violation state management

AGENT 3 — Frontend Upgrader

- Read DESIGN.md first. Follow it on every decision, no exceptions.
- Do NOT migrate to React, Next.js, or any framework
- Modularize all JS into ES modules: proctoring.js, auth.js, dashboard.js, api.js
- Integrate Tailwind CSS via CDN — use tailwind-design-system skill
- Apply the full DESIGN.md system: indigo canvas, Blurple primary, green/magenta accents,
  ABC Ginto Nord display type, gradient panels, rounded cards, full-bleed bands
- Redesign: login page, candidate instructions page, teacher dashboard
- Live violation feed on dashboard must feel real-time and visually urgent
- Do not break any existing functionality
- Ensure responsiveness and basic accessibility

AGENT 4 — Database & Auth

- Use sqlalchemy-postgres skill for all database work
- Replace any in-memory structured data with PostgreSQL via SQLAlchemy
- Create models for: User, Exam, Session, Violation, RiskScore
- Appwrite stays for video clip storage — do not touch it
- Implement proper JWT auth safe across multi-worker Gunicorn
- Add session invalidation on exam end or violation threshold breach

AGENT 5 — Testing & Docker

- Use pytest-coverage skill — write unit tests for risk scoring, cooldown logic, violation thresholds
- Use docker-expert skill — write a Dockerfile that correctly handles opencv-python-headless
  and MediaPipe dependencies for clean Railway deployment
- Update Procfile if needed for multi-worker Gunicorn with correct worker count
- Ensure .env variables are never hardcoded anywhere in the codebase

---

ORCHESTRATOR INSTRUCTIONS:

- Run all agents, collect all findings and changes
- Produce a final report: what was fixed, what was improved, what needs human decision
- If agents conflict, surface it — do not silently resolve
- Priority order: correctness → performance → structure → design
- Do not check in unless something is genuinely ambiguous or destructive

Start now.

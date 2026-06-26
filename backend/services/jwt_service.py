import os

JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "").strip()

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET_KEY is not set in environment")


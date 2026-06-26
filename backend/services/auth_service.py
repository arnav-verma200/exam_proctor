import hashlib


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


CANDIDATES: dict[str, dict] = {
    "240110012345": {"password": _hash("Pass@1234"), "name": "Arjun Mehta"},
    "240110056789": {"password": _hash("Pass@5678"), "name": "Priya Sharma"},
    "240110099001": {"password": _hash("Pass@9900"), "name": "Rahul Singh"},
}

TEACHERS: dict[str, str] = {
    "teacher1": _hash("Teacher@123"),
    "admin": _hash("Admin@2026"),
}


def authenticate_candidate(app_number: str, password: str) -> dict | None:
    cand = CANDIDATES.get(app_number)
    if cand and cand["password"] == _hash(password):
        return {"app_number": app_number, "name": cand["name"]}
    return None


def authenticate_teacher(username: str, password: str) -> str | None:
    stored = TEACHERS.get(username)
    if stored and stored == _hash(password):
        return username
    return None

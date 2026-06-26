"""
Seed script — populates the database with initial users and exam config.

Usage:
    python -m backend.seed

Safe to run multiple times (checks for existing records before inserting).
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.database import SessionLocal, init_db
from backend.models.db_models import Exam, User, UserRole


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


USERS = [
    {
        "app_number": "240110012345",
        "name": "Arjun Mehta",
        "password": "Pass@1234",
        "role": UserRole.CANDIDATE,
    },
    {
        "app_number": "240110056789",
        "name": "Priya Sharma",
        "password": "Pass@5678",
        "role": UserRole.CANDIDATE,
    },
    {
        "app_number": "240110099001",
        "name": "Rahul Singh",
        "password": "Pass@9900",
        "role": UserRole.CANDIDATE,
    },
    {
        "app_number": "teacher1",
        "name": "Teacher One",
        "password": "Teacher@123",
        "role": UserRole.TEACHER,
    },
    {
        "app_number": "admin",
        "name": "Admin",
        "password": "Admin@2026",
        "role": UserRole.TEACHER,
    },
]

EXAM = {
    "title": "JEE (Main) 2026",
    "paper": "Paper 1 (B.E. / B.Tech)",
    "duration_mins": 180,
    "total_questions": 75,
    "total_marks": 300,
}


def seed():
    init_db()
    db = SessionLocal()
    try:
        existing_users = db.query(User).count()
        if existing_users > 0:
            print(f"[seed] {existing_users} users already exist — skipping user seed.")
        else:
            for u in USERS:
                user = User(
                    app_number=u["app_number"],
                    name=u["name"],
                    password_hash=_hash(u["password"]),
                    role=u["role"],
                )
                db.add(user)
            db.flush()
            print(f"[seed] Inserted {len(USERS)} users.")

        existing_exams = db.query(Exam).count()
        if existing_exams > 0:
            print(f"[seed] {existing_exams} exams already exist — skipping exam seed.")
        else:
            exam = Exam(
                title=EXAM["title"],
                paper=EXAM["paper"],
                duration_mins=EXAM["duration_mins"],
                total_questions=EXAM["total_questions"],
                total_marks=EXAM["total_marks"],
            )
            db.add(exam)
            db.flush()
            print(f"[seed] Inserted exam: {exam.title}")

        db.commit()
        print("[seed] Seeding complete.")
    except Exception as e:
        db.rollback()
        print(f"[seed] Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()

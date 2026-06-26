import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text, DateTime, Enum, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class UserRole(str, enum.Enum):
    CANDIDATE = "candidate"
    TEACHER = "teacher"


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    SUBMITTED = "submitted"
    INVALIDATED = "invalidated"


class ViolationLevel(str, enum.Enum):
    WARN = "warn"
    DANGER = "danger"
    INFO = "info"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    app_number: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole), nullable=False, default=UserRole.CANDIDATE,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User(app_number={self.app_number}, role={self.role.value})>"


class Exam(Base):
    __tablename__ = "exams"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    paper: Mapped[str] = mapped_column(String(200), nullable=False)
    duration_mins: Mapped[int] = mapped_column(Integer, nullable=False)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    total_marks: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    sessions: Mapped[list["Session"]] = relationship(back_populates="exam")

    def __repr__(self) -> str:
        return f"<Exam(title={self.title})>"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    exam_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("exams.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), nullable=False, default=SessionStatus.ACTIVE,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    answers: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=dict)
    jwt_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="sessions")
    exam: Mapped[Optional["Exam"]] = relationship(back_populates="sessions")
    violations: Mapped[list["Violation"]] = relationship(
        back_populates="session", cascade="all, delete-orphan",
    )
    risk_score_record: Mapped[Optional["RiskScore"]] = relationship(
        back_populates="session", uselist=False, cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_sessions_user_status", "user_id", "status"),
        Index("ix_sessions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, status={self.status.value})>"


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    impact: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    level: Mapped[ViolationLevel] = mapped_column(
        Enum(ViolationLevel), nullable=False, default=ViolationLevel.INFO,
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snapshot_base64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    session: Mapped["Session"] = relationship(back_populates="violations")

    __table_args__ = (
        Index("ix_violations_session", "session_id"),
        Index("ix_violations_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Violation(event_type={self.event_type})>"


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    )

    session: Mapped["Session"] = relationship(back_populates="risk_score_record")

    def __repr__(self) -> str:
        return f"<RiskScore(session_id={self.session_id}, score={self.score})>"

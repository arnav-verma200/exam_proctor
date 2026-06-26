from pydantic import BaseModel, Field
from typing import Any, Optional


class LoginRequest(BaseModel):
    app_number: str = Field(..., description="Candidate application number")
    password: str = Field(..., description="Candidate password")


class TeacherLoginRequest(BaseModel):
    username: str = Field(..., description="Teacher username")
    password: str = Field(..., description="Teacher password")


class ExamStartRequest(BaseModel):
    agreed: bool = Field(..., description="Whether candidate agreed to instructions")


class SaveAnswerRequest(BaseModel):
    question_id: str = Field(..., description="Question identifier")
    answer: Any = Field(..., description="Answer value")


class SubmitRequest(BaseModel):
    answers: dict[str, Any] = Field(default_factory=dict, description="Final bulk answers")


class ProctorEventRequest(BaseModel):
    app_number: str = "UNKNOWN"
    event_type: str = "Unknown"
    confidence: int = 0
    impact: Optional[int] = None
    note: str = ""


class SnapshotRequest(BaseModel):
    app_number: str = ""
    student_name: str = ""
    event_type: str = ""
    image: str = Field(..., description="Base64 data-URI JPEG image")


class SetCandidateRequest(BaseModel):
    app_number: Optional[str] = None
    student_name: str = "Unknown"

from datetime import date, datetime
from pydantic import BaseModel

from app.models.leave import LeaveRequestStatus


class LeaveRequestCreate(BaseModel):
    section_id: int
    date: date
    reason: str | None = None


class LeaveRequestOut(BaseModel):
    id: int
    student_id: int
    student_name: str | None = None
    section_id: int
    section_name: str | None = None
    class_id: int
    class_name: str | None = None
    date: date
    reason: str | None
    status: LeaveRequestStatus
    requested_at: datetime

    class Config:
        from_attributes = True

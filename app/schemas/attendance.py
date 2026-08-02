from datetime import date, time, datetime
from pydantic import BaseModel

from app.models.attendance import AttendanceStatus


class AttendanceRecordIn(BaseModel):
    """One row the teacher confirms before saving — this is what the manual
    review/override screen posts. `student_id` is None for a still-unrecognized
    face the teacher couldn't match to a roster student (excluded from save)."""

    student_id: int
    status: AttendanceStatus


class AttendanceSessionCreate(BaseModel):
    section_id: int
    records: list[AttendanceRecordIn]


class AttendanceRecordOut(BaseModel):
    id: int
    student_id: int
    status: AttendanceStatus

    class Config:
        from_attributes = True


class AttendanceSessionOut(BaseModel):
    id: int
    section_id: int
    taken_by: int
    date: date
    time: time
    present_count: int
    absent_count: int
    leave_count: int
    created_at: datetime
    records: list[AttendanceRecordOut] = []

    class Config:
        from_attributes = True

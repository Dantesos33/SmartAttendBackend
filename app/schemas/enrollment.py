from datetime import datetime
from pydantic import BaseModel

from app.models.enrollment import EnrollmentRequestStatus


class EnrollmentRequestCreate(BaseModel):
    section_id: int


class EnrollmentRequestOut(BaseModel):
    id: int
    student_id: int
    student_name: str | None = None
    section_id: int
    section_name: str | None = None
    class_id: int
    class_name: str | None = None
    status: EnrollmentRequestStatus
    requested_at: datetime

    class Config:
        from_attributes = True


class EnrollmentOut(BaseModel):
    id: int
    student_id: int
    section_id: int
    class_id: int
    enrolled_at: datetime

    class Config:
        from_attributes = True

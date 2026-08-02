from datetime import time
from pydantic import BaseModel


class SectionCreate(BaseModel):
    name: str  # "A", "B", "C"
    schedule_days: str | None = None  # "Mon,Wed,Fri"
    start_time: time | None = None
    end_time: time | None = None


class SectionOut(BaseModel):
    id: int
    class_id: int
    name: str
    schedule_days: str | None
    start_time: time | None
    end_time: time | None
    student_count: int = 0

    class Config:
        from_attributes = True


class AddStudentInput(BaseModel):
    name: str
    student_id: str


class AddStudentsBulkRequest(BaseModel):
    section_id: int
    students: list[AddStudentInput]


class AddStudentResult(BaseModel):
    student_id: str
    status: str  # "created_and_enrolled" | "existing_enrolled" | "error"
    message: str


class ClassCreate(BaseModel):
    name: str
    code: str
    subject: str | None = None
    university_id: int
    # A class needs at least one section to actually be usable; allow creating
    # them together in one call so the frontend's "create class" form (which now
    # collects university + schedule up front) maps to a single request.
    sections: list[SectionCreate] = []


class ClassOut(BaseModel):
    id: int
    name: str
    code: str
    subject: str | None
    university_id: int
    university_name: str | None = None
    teacher_id: int
    sections: list[SectionOut] = []

    class Config:
        from_attributes = True


class SectionUpdate(BaseModel):
    id: int
    schedule_days: str | None = None
    start_time: time | None = None
    end_time: time | None = None


class ClassUpdate(BaseModel):
    name: str | None = None
    code: str | None = None
    subject: str | None = None
    sections: list[SectionUpdate] | None = None


class ClassBrowseOut(BaseModel):
    """What a student sees when browsing all classes across all teachers to
    request enrollment — deliberately excludes anything the owning teacher
    wouldn't want exposed (no student rosters, etc.)."""

    id: int
    name: str
    code: str
    subject: str | None
    university_id: int
    university_name: str | None = None
    teacher_name: str
    sections: list[SectionOut] = []
    already_enrolled_section_id: int | None = None  # set if the student is enrolled in *some* section of this class
    pending_request_section_id: int | None = None  # set if they have a pending request for this class

    class Config:
        from_attributes = True

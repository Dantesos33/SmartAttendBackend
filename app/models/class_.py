from sqlalchemy import Column, Integer, String, Time, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base


class Class(Base):
    """A course, e.g. 'English 1' — owned by exactly one teacher, belongs to one university.
    A teacher only ever sees classes where teacher_id == their own user id (enforced in the
    router layer via the current-user dependency, not just trusted from the client)."""

    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(190), nullable=False)
    code = Column(String(30), unique=True, nullable=False)  # e.g. "ENG101" — shown in the browse list
    subject = Column(String(190), nullable=True)
    university_id = Column(Integer, ForeignKey("universities.id"), nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    university = relationship("University", back_populates="classes")
    teacher = relationship("User", back_populates="owned_classes")
    sections = relationship("Section", back_populates="class_", cascade="all, delete-orphan")


class Section(Base):
    """A section of a class, e.g. 'A' / 'B' / 'C' for 'English 1'. Carries its own
    schedule (days + time) since different sections of the same class often meet
    at different times."""

    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    name = Column(String(50), nullable=False)  # "A", "B", "C"...
    schedule_days = Column(String(100), nullable=True)  # e.g. "Mon,Wed,Fri"
    start_time = Column(Time, nullable=True)
    end_time = Column(Time, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    class_ = relationship("Class", back_populates="sections")
    enrollments = relationship("Enrollment", back_populates="section", cascade="all, delete-orphan")
    attendance_sessions = relationship(
        "AttendanceSession", back_populates="section", cascade="all, delete-orphan"
    )

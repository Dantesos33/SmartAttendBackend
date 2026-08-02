import enum
import datetime # Added to generate safe date/time stamps natively via Python

from sqlalchemy import Column, Integer, Enum, Date, Time, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base


class AttendanceStatus(str, enum.Enum):
    present = "present"
    absent = "absent"
    leave = "leave"  # approved leave request covering this session


class AttendanceSession(Base):
    """One attendance-taking event for a section (one classroom photo processed by
    the AI recognition service). Manual overrides happen before this + its records
    are written, per the review step in the plan."""

    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)
    taken_by = Column(Integer, ForeignKey("users.id"), nullable=False)  # teacher
    
    # CHANGED: Swapped server_default for safe Python-level default execution blocks
    date = Column(Date, default=datetime.date.today, nullable=False)
    time = Column(Time, default=lambda: datetime.datetime.now().time(), nullable=False)
    
    present_count = Column(Integer, default=0)
    absent_count = Column(Integer, default=0)
    leave_count = Column(Integer, default=0)
    
    # DateTime functions work perfectly fine as server_defaults in all MySQL versions
    created_at = Column(DateTime, server_default=func.now())

    section = relationship("Section", back_populates="attendance_sessions")
    records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")


class AttendanceRecord(Base):
    """One student's present/absent result within a session. This is what
    triggers the per-student notification once the session is saved."""

    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(Enum(AttendanceStatus), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("AttendanceSession", back_populates="records")
    student = relationship("User")

import enum

from sqlalchemy import Column, Integer, String, Enum, DateTime, Boolean, Text, func
from sqlalchemy.orm import relationship

from app.database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    teacher = "teacher"
    student = "student"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(190), unique=True, index=True, nullable=True)
    # Students can log in with a student ID instead of email (e.g. when a
    # teacher adds them directly without collecting a personal email).
    # Teachers/admins always use email. Unique when set; NULL is allowed to
    # repeat (standard SQL behavior — multiple NULLs don't violate UNIQUE).
    student_id = Column(String(50), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.student)
    avatar_url = Column(String(500), nullable=True)
    # 128-d face embedding persisted as JSON — survives Railway redeploys.
    face_encoding_json = Column(Text, nullable=True)
    # True for accounts a teacher created directly (bulk or single add) with a
    # default password — forces a change-password gate on first login,
    # alongside the existing photo-verification gate.
    must_change_password = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # Universities a teacher belongs to (irrelevant for students/admin)
    universities = relationship(
        "TeacherUniversity", back_populates="teacher", cascade="all, delete-orphan"
    )
    owned_classes = relationship("Class", back_populates="teacher")
    enrollments = relationship("Enrollment", back_populates="student")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")

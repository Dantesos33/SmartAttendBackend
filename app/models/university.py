from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import relationship

from app.database import Base


class University(Base):
    __tablename__ = "universities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(190), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    classes = relationship("Class", back_populates="university")
    teachers = relationship("TeacherUniversity", back_populates="university")


class TeacherUniversity(Base):
    """A teacher can belong to multiple universities; a university has many teachers."""

    __tablename__ = "teacher_universities"
    __table_args__ = (UniqueConstraint("teacher_id", "university_id", name="uq_teacher_university"),)

    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    university_id = Column(Integer, ForeignKey("universities.id"), nullable=False)

    teacher = relationship("User", back_populates="universities")
    university = relationship("University", back_populates="teachers")

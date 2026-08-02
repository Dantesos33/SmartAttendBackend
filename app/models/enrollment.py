import enum

from sqlalchemy import (
    Column,
    Integer,
    Enum,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Enrollment(Base):
    """A student's confirmed enrollment in one section of one class.

    class_id is denormalized from section.class_id specifically so we can put a
    database-level UNIQUE constraint on (student_id, class_id) — this is what
    actually enforces "a student can't be enrolled in two sections of the same
    class" at the data layer, not just in application code."""

    __tablename__ = "enrollments"
    __table_args__ = (
        UniqueConstraint("student_id", "class_id", name="uq_student_one_section_per_class"),
    )

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    enrolled_at = Column(DateTime, server_default=func.now())

    student = relationship("User", back_populates="enrollments")
    section = relationship("Section", back_populates="enrollments")


class EnrollmentRequestStatus(str, enum.Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"


class EnrollmentRequest(Base):
    """A student's request to join a section — must be accepted by the owning
    teacher before an Enrollment row is created. A student may have at most one
    *pending* request per class (checked in the router), mirroring the same
    one-section-per-class rule that applies to confirmed enrollments."""

    __tablename__ = "enrollment_requests"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    status = Column(Enum(EnrollmentRequestStatus), default=EnrollmentRequestStatus.pending, nullable=False)
    requested_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)

    student = relationship("User")
    section = relationship("Section")

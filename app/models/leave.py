import enum

from sqlalchemy import Column, Integer, String, Date, Enum, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base


class LeaveRequestStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    date = Column(Date, nullable=False)
    reason = Column(String(500), nullable=True)
    status = Column(Enum(LeaveRequestStatus), default=LeaveRequestStatus.pending, nullable=False)
    requested_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)

    student = relationship("User")
    section = relationship("Section")

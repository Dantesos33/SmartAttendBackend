import enum

from sqlalchemy import Column, Integer, String, Text, Enum, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base


class NotificationType(str, enum.Enum):
    enrollment_accepted = "enrollment_accepted"
    enrollment_rejected = "enrollment_rejected"
    attendance_requested = "attendance_requested"  # student -> teacher ping
    attendance_result = "attendance_result"  # present/absent after AI processing
    enrollment_request_received = "enrollment_request_received"  # teacher sees new request
    low_attendance_warning = "low_attendance_warning"  # student's attendance just crossed below threshold
    enrolled_by_teacher = "enrolled_by_teacher"  # teacher directly added/enrolled this student
    leave_request_received = "leave_request_received"  # teacher sees new leave request
    leave_approved = "leave_approved"
    leave_rejected = "leave_rejected"


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(Enum(NotificationType), nullable=False)
    title = Column(String(190), nullable=False)
    body = Column(Text, nullable=True)
    related_id = Column(Integer, nullable=True)  # e.g. class_id, session_id — interpreted per `type`
    read_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="notifications")

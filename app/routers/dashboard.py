from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Class, Section
from app.models.enrollment import Enrollment, EnrollmentRequest, EnrollmentRequestStatus
from app.models.attendance import AttendanceSession, AttendanceRecord, AttendanceStatus
from app.models.notification import Notification
from app.core.deps import get_current_user
from app.routers.attendance import _class_attendance_percentage, LOW_ATTENDANCE_THRESHOLD

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
def dashboard_stats(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    unread_notifications = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id, Notification.read_at.is_(None))
        .count()
    )

    if current_user.role == UserRole.teacher:
        my_class_ids = [c.id for c in db.query(Class.id).filter(Class.teacher_id == current_user.id).all()]
        total_classes = len(my_class_ids)
        total_students = (
            db.query(Enrollment).filter(Enrollment.class_id.in_(my_class_ids)).count() if my_class_ids else 0
        )
        today_sessions = (
            db.query(AttendanceSession)
            .join(Section, AttendanceSession.section_id == Section.id)
            .filter(Section.class_id.in_(my_class_ids), AttendanceSession.date == date.today())
            .all()
            if my_class_ids
            else []
        )
        present_today = sum(s.present_count for s in today_sessions)
        absent_today = sum(s.absent_count for s in today_sessions)
        pending_requests = (
            db.query(EnrollmentRequest)
            .filter(
                EnrollmentRequest.class_id.in_(my_class_ids),
                EnrollmentRequest.status == EnrollmentRequestStatus.pending,
            )
            .count()
            if my_class_ids
            else 0
        )
        return {
            "role": "teacher",
            "total_classes": total_classes,
            "total_students": total_students,
            "present_today": present_today,
            "absent_today": absent_today,
            "pending_enrollment_requests": pending_requests,
            "unread_notifications": unread_notifications,
        }

    if current_user.role == UserRole.student:
        enrollments = db.query(Enrollment).filter(Enrollment.student_id == current_user.id).all()
        percentages = [
            _class_attendance_percentage(db, current_user.id, e.class_id) for e in enrollments
        ]
        percentages = [p for p in percentages if p is not None]
        overall = round(sum(percentages) / len(percentages), 1) if percentages else None
        pending_requests = (
            db.query(EnrollmentRequest)
            .filter(
                EnrollmentRequest.student_id == current_user.id,
                EnrollmentRequest.status == EnrollmentRequestStatus.pending,
            )
            .count()
        )
        return {
            "role": "student",
            "enrolled_classes": len(enrollments),
            "overall_attendance_percentage": overall,
            "pending_enrollment_requests": pending_requests,
            "unread_notifications": unread_notifications,
        }

    # admin
    total_students = db.query(User).filter(User.role == UserRole.student).count()
    total_teachers = db.query(User).filter(User.role == UserRole.teacher).count()
    total_classes = db.query(Class).count()
    today_sessions = db.query(AttendanceSession).filter(AttendanceSession.date == date.today()).all()
    present_today = sum(s.present_count for s in today_sessions)
    absent_today = sum(s.absent_count for s in today_sessions)

    low_attendance_count = 0
    for enrollment in db.query(Enrollment).all():
        pct = _class_attendance_percentage(db, enrollment.student_id, enrollment.class_id)
        if pct is not None and pct < LOW_ATTENDANCE_THRESHOLD:
            low_attendance_count += 1

    return {
        "role": "admin",
        "total_students": total_students,
        "total_teachers": total_teachers,
        "total_classes": total_classes,
        "present_today": present_today,
        "absent_today": absent_today,
        "low_attendance_count": low_attendance_count,
        "unread_notifications": unread_notifications,
    }

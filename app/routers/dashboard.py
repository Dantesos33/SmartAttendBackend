from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import case, func

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Class, Section
from app.models.enrollment import Enrollment, EnrollmentRequest, EnrollmentRequestStatus
from app.models.attendance import AttendanceSession, AttendanceRecord, AttendanceStatus
from app.models.notification import Notification
from app.core.deps import get_current_user
from app.routers.attendance import LOW_ATTENDANCE_THRESHOLD

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
        today_totals = (
            db.query(
                func.coalesce(func.sum(AttendanceSession.present_count), 0),
                func.coalesce(func.sum(AttendanceSession.absent_count), 0),
            )
            .join(Section, AttendanceSession.section_id == Section.id)
            .filter(Section.class_id.in_(my_class_ids), AttendanceSession.date == date.today())
            .one()
            if my_class_ids
            else (0, 0)
        )
        present_today, absent_today = today_totals
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
        enrolled_class_ids = [
            row[0]
            for row in db.query(Enrollment.class_id)
            .filter(Enrollment.student_id == current_user.id)
            .all()
        ]
        attendance_totals = (
            db.query(
                Section.class_id,
                func.sum(case((AttendanceRecord.status == AttendanceStatus.present, 1), else_=0)).label("present_count"),
                func.count(AttendanceRecord.id).label("record_count"),
            )
            .join(AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id)
            .join(Section, AttendanceSession.section_id == Section.id)
            .filter(
                AttendanceRecord.student_id == current_user.id,
                AttendanceRecord.status != AttendanceStatus.leave,
                Section.class_id.in_(enrolled_class_ids),
            )
            .group_by(Section.class_id)
            .all()
            if enrolled_class_ids
            else []
        )
        percentages = [
            (row.present_count / row.record_count) * 100
            for row in attendance_totals
            if row.record_count
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
            "enrolled_classes": len(enrolled_class_ids),
            "overall_attendance_percentage": overall,
            "pending_enrollment_requests": pending_requests,
            "unread_notifications": unread_notifications,
        }

    # admin
    total_students = db.query(User).filter(User.role == UserRole.student).count()
    total_teachers = db.query(User).filter(User.role == UserRole.teacher).count()
    total_classes = db.query(Class).count()
    today_totals = (
        db.query(
            func.coalesce(func.sum(AttendanceSession.present_count), 0),
            func.coalesce(func.sum(AttendanceSession.absent_count), 0),
        )
        .filter(AttendanceSession.date == date.today())
        .one()
    )
    present_today, absent_today = today_totals

    attendance_totals = (
        db.query(
            AttendanceRecord.student_id,
            Section.class_id,
            func.sum(
                case(
                    (AttendanceRecord.status == AttendanceStatus.present, 1),
                    else_=0,
                )
            ).label("present_count"),
            func.count(AttendanceRecord.id).label("record_count"),
        )
        .join(AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id)
        .join(Section, AttendanceSession.section_id == Section.id)
        .filter(AttendanceRecord.status != AttendanceStatus.leave)
        .group_by(AttendanceRecord.student_id, Section.class_id)
        .all()
    )
    low_attendance_count = sum(
        1
        for row in attendance_totals
        if row.record_count
        and (row.present_count / row.record_count) * 100 < LOW_ATTENDANCE_THRESHOLD
    )

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

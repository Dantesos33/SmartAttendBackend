from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Section, Class
from app.models.enrollment import Enrollment
from app.models.attendance import AttendanceSession, AttendanceRecord, AttendanceStatus
from app.schemas.attendance import AttendanceSessionCreate, AttendanceSessionOut
from app.core.deps import get_current_user, require_role
from app.core.notifications import notify
from app.models.notification import NotificationType

router = APIRouter(prefix="/attendance", tags=["attendance"])


LOW_ATTENDANCE_THRESHOLD = 75.0


def _class_attendance_percentage(db: Session, student_id: int, class_id: int, exclude_session_id: int | None = None) -> float | None:
    """Present / (present + absent) across every attendance record for this
    student within this class — "leave" records are excused and excluded
    from both the numerator and denominator entirely, so an approved leave
    day neither helps nor hurts the percentage. Optionally excludes one
    session (used to compute the before-this-session baseline). Returns
    None if there's no countable history yet."""
    query = (
        db.query(AttendanceRecord)
        .join(AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id)
        .join(Section, AttendanceSession.section_id == Section.id)
        .filter(
            AttendanceRecord.student_id == student_id,
            Section.class_id == class_id,
            AttendanceRecord.status != AttendanceStatus.leave,
        )
    )
    if exclude_session_id is not None:
        query = query.filter(AttendanceRecord.session_id != exclude_session_id)
    records = query.all()
    if not records:
        return None
    present = sum(1 for r in records if r.status == AttendanceStatus.present)
    return (present / len(records)) * 100


@router.post("/sessions", response_model=AttendanceSessionOut, status_code=201)
def save_attendance_session(
    payload: AttendanceSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """This is where the manually-reviewed attendance result actually gets
    committed — the AI recognition service returns raw results first, the
    teacher reviews/overrides on the frontend, and THIS endpoint is only called
    once with the final, confirmed list. Every enrolled student not present in
    `records` at all still needs an explicit status — the frontend is expected
    to send a row for every enrolled student (present or absent), not just the
    recognized ones, so nobody is silently left out of the session."""
    section = db.query(Section).filter(Section.id == payload.section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")
    if section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")

    present_count = sum(1 for r in payload.records if r.status == AttendanceStatus.present)
    absent_count = sum(1 for r in payload.records if r.status == AttendanceStatus.absent)
    leave_count = sum(1 for r in payload.records if r.status == AttendanceStatus.leave)

    session_row = AttendanceSession(
        section_id=section.id,
        taken_by=current_user.id,
        present_count=present_count,
        absent_count=absent_count,
        leave_count=leave_count,
    )
    db.add(session_row)
    db.flush()

    for r in payload.records:
        db.add(AttendanceRecord(session_id=session_row.id, student_id=r.student_id, status=r.status))
        notify(
            db,
            user_id=r.student_id,
            type_=NotificationType.attendance_result,
            title=f"You were marked {r.status.value.capitalize()} in {section.class_.name}",
            body=f"Section {section.name} — {session_row.date or date.today()}",
            related_id=session_row.id,
        )

    db.flush()  # so the percentage query below can see the just-added records

    # Proactive low-attendance warning: only fire the moment a student's running
    # percentage CROSSES below the threshold, not on every session while they
    # stay chronically low — otherwise they'd get spammed every single class.
    for r in payload.records:
        before = _class_attendance_percentage(db, r.student_id, section.class_id, exclude_session_id=session_row.id)
        after = _class_attendance_percentage(db, r.student_id, section.class_id)
        if before is not None and before >= LOW_ATTENDANCE_THRESHOLD and after is not None and after < LOW_ATTENDANCE_THRESHOLD:
            notify(
                db,
                user_id=r.student_id,
                type_=NotificationType.low_attendance_warning,
                title=f"Your attendance in {section.class_.name} has dropped below {int(LOW_ATTENDANCE_THRESHOLD)}%",
                body=f"Current attendance: {after:.1f}%. Please check with your teacher if this seems wrong.",
                related_id=section.class_id,
            )

    db.commit()
    db.refresh(session_row)
    return session_row


@router.get("/low-attendance")
def low_attendance_students(
    threshold: float = LOW_ATTENDANCE_THRESHOLD,
    university_id: int | None = None,
    class_id: int | None = None,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(UserRole.admin)),
):
    """Admin's low-attendance filter — every (student, class) pair currently
    under the threshold, optionally narrowed to one university or class."""
    from app.models.class_ import Class as ClassModel

    query = db.query(Enrollment).join(ClassModel, Enrollment.class_id == ClassModel.id)
    if university_id:
        query = query.filter(ClassModel.university_id == university_id)
    if class_id:
        query = query.filter(Enrollment.class_id == class_id)

    results = []
    for enrollment in query.all():
        pct = _class_attendance_percentage(db, enrollment.student_id, enrollment.class_id)
        if pct is not None and pct < threshold:
            results.append(
                {
                    "student_id": enrollment.student_id,
                    "student_name": enrollment.student.name,
                    "class_id": enrollment.class_id,
                    "class_name": enrollment.section.class_.name,
                    "attendance_percentage": round(pct, 1),
                }
            )
    return results


@router.get("/student/{student_id}/history")
def student_attendance_history(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """A single student's attendance records across every class they're
    enrolled in — student-centric, unlike /sessions/section/{id} which is
    section-centric. A student may only view their own; a teacher may only
    view students enrolled in one of their own classes."""
    if current_user.role == UserRole.student and current_user.id != student_id:
        raise HTTPException(status_code=403, detail="You can only view your own attendance history.")

    if current_user.role == UserRole.teacher:
        owns_a_class_with_this_student = (
            db.query(Enrollment)
            .join(Section, Enrollment.section_id == Section.id)
            .join(Class, Section.class_id == Class.id)
            .filter(Enrollment.student_id == student_id, Class.teacher_id == current_user.id)
            .first()
        )
        if not owns_a_class_with_this_student:
            raise HTTPException(status_code=403, detail="This student isn't enrolled in any of your classes.")

    records = (
        db.query(AttendanceRecord)
        .join(AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id)
        .join(Section, AttendanceSession.section_id == Section.id)
        .join(Class, Section.class_id == Class.id)
        .options(joinedload(AttendanceRecord.session))
        .filter(AttendanceRecord.student_id == student_id)
        .order_by(AttendanceSession.date.desc(), AttendanceSession.time.desc())
        .all()
    )

    total_countable = sum(1 for r in records if r.status != AttendanceStatus.leave)
    present = sum(1 for r in records if r.status == AttendanceStatus.present)
    absent = sum(1 for r in records if r.status == AttendanceStatus.absent)
    leave = sum(1 for r in records if r.status == AttendanceStatus.leave)

    return {
        "student_id": student_id,
        "total_sessions": len(records),
        "present_count": present,
        "absent_count": absent,
        "leave_count": leave,
        "attendance_percentage": round((present / total_countable) * 100, 1) if total_countable else None,
        "records": [
            {
                "session_id": r.session_id,
                "class_name": r.session.section.class_.name,
                "section_name": r.session.section.name,
                "date": str(r.session.date),
                "time": str(r.session.time),
                "status": r.status.value,
            }
            for r in records
        ],
    }


@router.get("/sessions/section/{section_id}", response_model=list[AttendanceSessionOut])
def section_history(
    section_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    section = db.query(Section).filter(Section.id == section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")

    if current_user.role == UserRole.teacher and section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")
    if current_user.role == UserRole.student:
        enrolled = (
            db.query(Enrollment)
            .filter(Enrollment.student_id == current_user.id, Enrollment.section_id == section_id)
            .first()
        )
        if not enrolled:
            raise HTTPException(status_code=403, detail="You're not enrolled in this section.")

    return (
        db.query(AttendanceSession)
        .options(joinedload(AttendanceSession.records))
        .filter(AttendanceSession.section_id == section_id)
        .order_by(AttendanceSession.date.desc(), AttendanceSession.time.desc())
        .all()
    )


@router.post("/request-taking/{section_id}", status_code=201)
def request_attendance_taking(
    section_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.student)),
):
    """A student pings their teacher to take attendance if it hasn't happened
    yet today for a section they're enrolled in."""
    enrolled = (
        db.query(Enrollment)
        .filter(Enrollment.student_id == current_user.id, Enrollment.section_id == section_id)
        .first()
    )
    if not enrolled:
        raise HTTPException(status_code=403, detail="You're not enrolled in this section.")

    section = db.query(Section).filter(Section.id == section_id).first()

    already_taken_today = (
        db.query(AttendanceSession)
        .filter(AttendanceSession.section_id == section_id, AttendanceSession.date == date.today())
        .first()
    )
    if already_taken_today:
        raise HTTPException(status_code=409, detail="Attendance was already taken for this section today.")

    notify(
        db,
        user_id=section.class_.teacher_id,
        type_=NotificationType.attendance_requested,
        title=f"{current_user.name} requested attendance for {section.class_.name} ({section.name})",
        related_id=section.id,
    )
    db.commit()
    return {"status": "success", "message": "Request sent to your teacher."}

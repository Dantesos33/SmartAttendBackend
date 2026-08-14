from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import case, func

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Section, Class
from app.models.enrollment import Enrollment
from app.models.attendance import AttendanceSession, AttendanceRecord, AttendanceStatus
from app.schemas.attendance import AttendanceSessionCreate, AttendanceSessionOut
from app.core.deps import get_current_user, require_role
from app.core.notifications import notify
from app.core.db_helpers import get_section_with_class
from app.models.notification import NotificationType

router = APIRouter(prefix="/attendance", tags=["attendance"])


LOW_ATTENDANCE_THRESHOLD = 75.0


def _attendance_totals_by_student(
    db: Session,
    student_ids: list[int],
    class_id: int,
    exclude_session_id: int | None = None,
) -> dict[int, tuple[int, int]]:
    query = (
        db.query(
            AttendanceRecord.student_id,
            func.sum(case((AttendanceRecord.status == AttendanceStatus.present, 1), else_=0)).label("present_count"),
            func.count(AttendanceRecord.id).label("record_count"),
        )
        .join(AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id)
        .join(Section, AttendanceSession.section_id == Section.id)
        .filter(
            AttendanceRecord.student_id.in_(student_ids),
            Section.class_id == class_id,
            AttendanceRecord.status != AttendanceStatus.leave,
        )
        .group_by(AttendanceRecord.student_id)
    )
    if exclude_session_id is not None:
        query = query.filter(AttendanceRecord.session_id != exclude_session_id)
    return {row.student_id: (row.present_count, row.record_count) for row in query.all()}


@router.post("/sessions", response_model=AttendanceSessionOut)
def save_attendance_session(
    payload: AttendanceSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """Save or update today's attendance for a section — only one session per
    section per calendar day. A second save the same day replaces the records."""
    section = get_section_with_class(db, payload.section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")
    if section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")

    enrolled_ids = {
        row.student_id
        for row in db.query(Enrollment.student_id)
        .filter(Enrollment.section_id == section.id)
        .all()
    }
    for record in payload.records:
        if record.student_id not in enrolled_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Student {record.student_id} is not enrolled in this section.",
            )

    present_count = sum(1 for r in payload.records if r.status == AttendanceStatus.present)
    absent_count = sum(1 for r in payload.records if r.status == AttendanceStatus.absent)
    leave_count = sum(1 for r in payload.records if r.status == AttendanceStatus.leave)

    today = date.today()
    existing = (
        db.query(AttendanceSession)
        .options(joinedload(AttendanceSession.records))
        .filter(AttendanceSession.section_id == section.id, AttendanceSession.date == today)
        .first()
    )

    old_status_by_student: dict[int, AttendanceStatus] = {}
    if existing:
        session_row = existing
        old_status_by_student = {r.student_id: r.status for r in existing.records}
        db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_row.id).delete()
        session_row.present_count = present_count
        session_row.absent_count = absent_count
        session_row.leave_count = leave_count
        session_row.time = datetime.now().time()
        session_row.taken_by = current_user.id
    else:
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
        previous = old_status_by_student.get(r.student_id)
        if previous != r.status:
            verb = "updated to" if previous is not None else "marked"
            notify(
                db,
                user_id=r.student_id,
                type_=NotificationType.attendance_result,
                title=f"You were {verb} {r.status.value.capitalize()} in {section.class_.name}",
                body=f"Section {section.name} — {session_row.date or today}",
                related_id=session_row.id,
            )

    db.flush()

    student_ids = list({record.student_id for record in payload.records})
    before_totals = _attendance_totals_by_student(db, student_ids, section.class_id, session_row.id)
    after_totals = _attendance_totals_by_student(db, student_ids, section.class_id)
    for student_id in student_ids:
        before_present, before_count = before_totals.get(student_id, (0, 0))
        after_present, after_count = after_totals.get(student_id, (0, 0))
        before = (before_present / before_count) * 100 if before_count else None
        after = (after_present / after_count) * 100 if after_count else None
        if before is not None and before >= LOW_ATTENDANCE_THRESHOLD and after is not None and after < LOW_ATTENDANCE_THRESHOLD:
            notify(
                db,
                user_id=student_id,
                type_=NotificationType.low_attendance_warning,
                title=f"Your attendance in {section.class_.name} has dropped below {int(LOW_ATTENDANCE_THRESHOLD)}%",
                body=f"Current attendance: {after:.1f}%. Please check with your teacher if this seems wrong.",
                related_id=section.class_id,
            )

    db.commit()
    db.refresh(session_row)
    return session_row


@router.get("/sessions/section/{section_id}/today", response_model=AttendanceSessionOut | None)
def get_today_session(
    section_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    section = get_section_with_class(db, section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")
    if section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")

    session = (
        db.query(AttendanceSession)
        .options(joinedload(AttendanceSession.records))
        .filter(AttendanceSession.section_id == section_id, AttendanceSession.date == date.today())
        .first()
    )
    # No attendance today is a normal state, not an API error. Returning null
    # lets the Take Attendance screen render immediately without showing a
    # loading/error state just because the teacher has not taken attendance yet.
    return session


@router.delete("/sessions/{session_id}", status_code=204)
def delete_attendance_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    session = (
        db.query(AttendanceSession)
        .options(joinedload(AttendanceSession.section).joinedload(Section.class_))
        .filter(AttendanceSession.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Attendance session not found.")
    if session.section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")

    db.delete(session)
    db.commit()
    return None


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

    attendance_totals = (
        db.query(
            AttendanceRecord.student_id.label("student_id"),
            Section.class_id.label("class_id"),
            func.sum(
                case((AttendanceRecord.status == AttendanceStatus.present, 1), else_=0)
            ).label("present_count"),
            func.count(AttendanceRecord.id).label("record_count"),
        )
        .join(AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id)
        .join(Section, AttendanceSession.section_id == Section.id)
        .filter(AttendanceRecord.status != AttendanceStatus.leave)
        .group_by(AttendanceRecord.student_id, Section.class_id)
        .subquery()
    )

    query = (
        db.query(
            Enrollment.student_id,
            User.name.label("student_name"),
            Enrollment.class_id,
            ClassModel.name.label("class_name"),
            attendance_totals.c.present_count,
            attendance_totals.c.record_count,
        )
        .join(User, Enrollment.student_id == User.id)
        .join(ClassModel, Enrollment.class_id == ClassModel.id)
        .outerjoin(
            attendance_totals,
            (attendance_totals.c.student_id == Enrollment.student_id)
            & (attendance_totals.c.class_id == Enrollment.class_id),
        )
    )
    if university_id:
        query = query.filter(ClassModel.university_id == university_id)
    if class_id:
        query = query.filter(Enrollment.class_id == class_id)

    results = []
    for row in query.all():
        if row.record_count:
            pct = (row.present_count / row.record_count) * 100
        else:
            pct = None
        if pct is not None and pct < threshold:
            results.append(
                {
                    "student_id": row.student_id,
                    "student_name": row.student_name,
                    "class_id": row.class_id,
                    "class_name": row.class_name,
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
        .options(
            joinedload(AttendanceRecord.session)
            .joinedload(AttendanceSession.section)
            .joinedload(Section.class_)
        )
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
    section = get_section_with_class(db, section_id)
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

    section = get_section_with_class(db, section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")

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

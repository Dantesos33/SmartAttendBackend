from datetime import datetime, date

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Section
from app.models.enrollment import Enrollment
from app.models.leave import LeaveRequest, LeaveRequestStatus
from app.models.attendance import AttendanceSession, AttendanceRecord, AttendanceStatus
from app.schemas.leave import LeaveRequestCreate, LeaveRequestOut
from app.core.deps import get_current_user, require_role
from app.core.notifications import notify
from app.core.db_helpers import get_section_with_class
from app.models.notification import NotificationType

router = APIRouter(prefix="/leave-requests", tags=["leave-requests"])


def _refresh_session_counts(db: Session, session: AttendanceSession) -> None:
    records = (
        db.query(AttendanceRecord)
        .filter(AttendanceRecord.session_id == session.id)
        .all()
    )
    session.present_count = sum(1 for r in records if r.status == AttendanceStatus.present)
    session.absent_count = sum(1 for r in records if r.status == AttendanceStatus.absent)
    session.leave_count = sum(1 for r in records if r.status == AttendanceStatus.leave)


def _record_leave_attendance(
    db: Session,
    lr: LeaveRequest,
    status: AttendanceStatus,
    teacher_id: int,
) -> AttendanceSession:
    """When a teacher resolves a leave request, write present/absent for that date."""
    session = (
        db.query(AttendanceSession)
        .filter(
            AttendanceSession.section_id == lr.section_id,
            AttendanceSession.date == lr.date,
        )
        .first()
    )
    if not session:
        session = AttendanceSession(
            section_id=lr.section_id,
            taken_by=teacher_id,
            date=lr.date,
            present_count=0,
            absent_count=0,
            leave_count=0,
        )
        db.add(session)
        db.flush()

    record = (
        db.query(AttendanceRecord)
        .filter(
            AttendanceRecord.session_id == session.id,
            AttendanceRecord.student_id == lr.student_id,
        )
        .first()
    )
    if record:
        record.status = status
    else:
        db.add(
            AttendanceRecord(
                session_id=session.id,
                student_id=lr.student_id,
                status=status,
            )
        )
        notify(
            db,
            user_id=lr.student_id,
            type_=NotificationType.attendance_result,
            title=f"You were marked {status.value.capitalize()} in {lr.section.class_.name}",
            body=f"Section {lr.section.name} — {lr.date}",
            related_id=session.id,
        )

    db.flush()
    _refresh_session_counts(db, session)
    return session


def _to_out(lr: LeaveRequest) -> LeaveRequestOut:
    return LeaveRequestOut(
        id=lr.id,
        student_id=lr.student_id,
        student_name=lr.student.name if lr.student else None,
        section_id=lr.section_id,
        section_name=lr.section.name if lr.section else None,
        class_id=lr.class_id,
        class_name=lr.section.class_.name if lr.section else None,
        date=lr.date,
        reason=lr.reason,
        status=lr.status,
        requested_at=lr.requested_at,
    )


@router.post("", response_model=LeaveRequestOut, status_code=201)
def request_leave(
    payload: LeaveRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.student)),
):
    section = get_section_with_class(db, payload.section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")

    enrolled = (
        db.query(Enrollment)
        .filter(Enrollment.student_id == current_user.id, Enrollment.section_id == section.id)
        .first()
    )
    if not enrolled:
        raise HTTPException(status_code=403, detail="You're not enrolled in this section.")

    existing = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.student_id == current_user.id,
            LeaveRequest.section_id == section.id,
            LeaveRequest.date == payload.date,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="You've already requested leave for this date.")

    lr = LeaveRequest(
        student_id=current_user.id,
        section_id=section.id,
        class_id=section.class_id,
        date=payload.date,
        reason=payload.reason,
    )
    db.add(lr)
    db.flush()

    notify(
        db,
        user_id=section.class_.teacher_id,
        type_=NotificationType.leave_request_received,
        title=f"{current_user.name} requested leave for {section.class_.name} ({section.name})",
        body=f"Date: {payload.date}" + (f" — {payload.reason}" if payload.reason else ""),
        related_id=lr.id,
    )

    db.commit()
    db.refresh(lr)
    return _to_out(lr)


@router.get("/mine", response_model=list[LeaveRequestOut])
def my_leave_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.student)),
):
    requests = (
        db.query(LeaveRequest)
        .options(
            joinedload(LeaveRequest.student),
            joinedload(LeaveRequest.section).joinedload(Section.class_),
        )
        .filter(LeaveRequest.student_id == current_user.id)
        .order_by(LeaveRequest.requested_at.desc())
        .all()
    )
    return [_to_out(r) for r in requests]


@router.get("/pending", response_model=list[LeaveRequestOut])
def pending_leave_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    requests = (
        db.query(LeaveRequest)
        .join(Section, LeaveRequest.section_id == Section.id)
        .options(
            joinedload(LeaveRequest.student),
            joinedload(LeaveRequest.section).joinedload(Section.class_),
        )
        .filter(
            Section.class_.has(teacher_id=current_user.id),
            LeaveRequest.status == LeaveRequestStatus.pending,
        )
        .all()
    )
    return [_to_out(r) for r in requests]


def _get_owned_leave_request(db: Session, request_id: int, teacher: User) -> LeaveRequest:
    lr = (
        db.query(LeaveRequest)
        .options(
            joinedload(LeaveRequest.section).joinedload(Section.class_),
            joinedload(LeaveRequest.student),
        )
        .filter(LeaveRequest.id == request_id)
        .first()
    )
    if not lr:
        raise HTTPException(status_code=404, detail="Leave request not found.")
    if lr.section.class_.teacher_id != teacher.id:
        raise HTTPException(status_code=403, detail="This request isn't for one of your classes.")
    if lr.status != LeaveRequestStatus.pending:
        raise HTTPException(status_code=409, detail=f"Request already {lr.status.value}.")
    return lr


@router.post("/{request_id}/accept", response_model=LeaveRequestOut)
def accept_leave_request(
    request_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """Teacher marks the student Present for the requested leave date."""
    lr = _get_owned_leave_request(db, request_id, current_user)
    lr.status = LeaveRequestStatus.approved
    lr.resolved_at = datetime.utcnow()
    _record_leave_attendance(db, lr, AttendanceStatus.present, current_user.id)

    notify(
        db,
        user_id=lr.student_id,
        type_=NotificationType.leave_approved,
        title=f"Leave approved — marked Present for {lr.section.class_.name}",
        body=f"Section {lr.section.name} on {lr.date}",
        related_id=lr.id,
    )

    db.commit()
    db.refresh(lr)
    return _to_out(lr)


@router.post("/{request_id}/reject", response_model=LeaveRequestOut)
def reject_leave_request(
    request_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """Teacher marks the student Absent for the requested leave date."""
    lr = _get_owned_leave_request(db, request_id, current_user)
    lr.status = LeaveRequestStatus.rejected
    lr.resolved_at = datetime.utcnow()
    _record_leave_attendance(db, lr, AttendanceStatus.absent, current_user.id)

    notify(
        db,
        user_id=lr.student_id,
        type_=NotificationType.leave_rejected,
        title=f"Leave declined — marked Absent for {lr.section.class_.name}",
        body=f"Section {lr.section.name} on {lr.date}",
        related_id=lr.id,
    )

    db.commit()
    db.refresh(lr)
    return _to_out(lr)


@router.get("/approved-for-section/{section_id}")
def approved_leaves_for_section(
    section_id: int,
    date: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """Used by the attendance-review screen to pre-fill a student's status as
    'leave' instead of 'absent' when they have an approved leave for that
    exact date."""
    section = get_section_with_class(db, section_id)
    if not section or section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")

    approved = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.section_id == section_id,
            LeaveRequest.date == date,
            LeaveRequest.status == LeaveRequestStatus.approved,
        )
        .all()
    )
    return [lr.student_id for lr in approved]

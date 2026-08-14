from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload, selectinload
from datetime import time

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Class, Section
from app.models.university import University
from app.models.enrollment import Enrollment, EnrollmentRequest, EnrollmentRequestStatus
from app.schemas.class_ import ClassCreate, ClassUpdate, ClassOut, ClassBrowseOut, SectionOut
from app.core.deps import get_current_user, require_role
from app.core.db_helpers import load_class_for_out

router = APIRouter(prefix="/classes", tags=["classes"])


def _class_to_out(cls: Class) -> ClassOut:
    return ClassOut(
        id=cls.id,
        name=cls.name,
        code=cls.code,
        subject=cls.subject,
        university_id=cls.university_id,
        university_name=cls.university.name if cls.university else None,
        teacher_id=cls.teacher_id,
        sections=[
            SectionOut(
                id=s.id,
                class_id=s.class_id,
                name=s.name,
                schedule_days=s.schedule_days,
                start_time=s.start_time,
                end_time=s.end_time,
                student_count=len(s.enrollments),
            )
            for s in cls.sections
        ],
    )


def _validate_schedule(
    db: Session,
    teacher_id: int,
    proposed: list[tuple[str | None, time | None, time | None]],
    exclude_class_id: int | None = None,
) -> None:
    """Reject invalid times and overlapping sections for one teacher."""
    def days(value: str | None) -> set[str]:
        return {day.strip().lower() for day in (value or "").split(",") if day.strip()}

    for schedule_days, start, end in proposed:
        if start is None or end is None:
            continue
        if start >= end:
            raise HTTPException(status_code=400, detail="Class end time must be after start time.")

    query = (
        db.query(Section)
        .join(Class, Section.class_id == Class.id)
        .filter(Class.teacher_id == teacher_id)
    )
    if exclude_class_id is not None:
        query = query.filter(Class.id != exclude_class_id)

    existing = query.all()
    for proposed_days, proposed_start, proposed_end in proposed:
        proposed_day_set = days(proposed_days)
        if proposed_start is None or proposed_end is None:
            continue
        for section in existing:
            if section.start_time is None or section.end_time is None:
                continue
            if not proposed_day_set.intersection(days(section.schedule_days)):
                continue
            if proposed_start < section.end_time and section.start_time < proposed_end:
                raise HTTPException(
                    status_code=409,
                    detail="This teacher already has a class scheduled during that time.",
                )


@router.post("", response_model=ClassOut, status_code=201)
def create_class(
    payload: ClassCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """A class is created with its university and schedule up front (per the
    finalized requirement — no more 'TBD' placeholder), and is always owned by
    the teacher creating it."""
    university = db.query(University).filter(University.id == payload.university_id).first()
    if not university:
        raise HTTPException(status_code=404, detail="University not found.")

    _validate_schedule(
        db,
        current_user.id,
        [(s.schedule_days, s.start_time, s.end_time) for s in payload.sections],
    )

    existing_code = db.query(Class).filter(Class.code == payload.code).first()
    if existing_code:
        raise HTTPException(status_code=400, detail=f"Class code '{payload.code}' is already in use.")

    cls = Class(
        name=payload.name,
        code=payload.code,
        subject=payload.subject,
        university_id=payload.university_id,
        teacher_id=current_user.id,
    )
    db.add(cls)
    db.flush()  # get cls.id before creating sections

    for section_in in payload.sections:
        db.add(
            Section(
                class_id=cls.id,
                name=section_in.name,
                schedule_days=section_in.schedule_days,
                start_time=section_in.start_time,
                end_time=section_in.end_time,
            )
        )

    db.commit()
    return _class_to_out(load_class_for_out(db, cls.id))


@router.get("/enrolled", response_model=list[ClassOut])
def my_enrolled_classes(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.student)),
):
    """A student's own enrolled classes with full details (name, code,
    schedule) — unlike /enrollments/mine which only returns bare enrollment
    records, this is what the student dashboard actually needs to render."""
    class_ids = [
        e.class_id for e in db.query(Enrollment).filter(Enrollment.student_id == current_user.id).all()
    ]
    if not class_ids:
        return []
    classes = (
        db.query(Class)
        .options(
            joinedload(Class.university),
            joinedload(Class.sections).selectinload(Section.enrollments),
        )
        .filter(Class.id.in_(class_ids))
        .all()
    )
    return [_class_to_out(c) for c in classes]


@router.get("/mine", response_model=list[ClassOut])
def my_classes(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """A teacher only ever sees classes they created — ownership filter, not
    just university membership."""
    classes = (
        db.query(Class)
        .options(
            joinedload(Class.university),
            joinedload(Class.sections).selectinload(Section.enrollments),
        )
        .filter(Class.teacher_id == current_user.id)
        .all()
    )
    return [_class_to_out(c) for c in classes]


@router.get("/browse", response_model=list[ClassBrowseOut])
def browse_classes(
    university_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.student)),
):
    """A student browsing sees classes from ALL teachers (optionally filtered by
    university), each flagged with whether they're already enrolled or have a
    pending request for that class — so the frontend can grey out / relabel the
    request button instead of allowing a second request for the same class."""
    query = db.query(Class).options(
        joinedload(Class.university),
        joinedload(Class.teacher),
        joinedload(Class.sections).selectinload(Section.enrollments),
    )
    if university_id:
        query = query.filter(Class.university_id == university_id)
    classes = query.all()

    my_enrollments = {
        e.class_id: e.section_id
        for e in db.query(Enrollment).filter(Enrollment.student_id == current_user.id).all()
    }
    my_pending = {
        r.class_id: r.section_id
        for r in db.query(EnrollmentRequest)
        .filter(
            EnrollmentRequest.student_id == current_user.id,
            EnrollmentRequest.status == EnrollmentRequestStatus.pending,
        )
        .all()
    }

    out = []
    for cls in classes:
        out.append(
            ClassBrowseOut(
                id=cls.id,
                name=cls.name,
                code=cls.code,
                subject=cls.subject,
                university_id=cls.university_id,
                university_name=cls.university.name if cls.university else None,
                teacher_name=cls.teacher.name if cls.teacher else "Unknown",
                sections=[
                    SectionOut(
                        id=s.id,
                        class_id=s.class_id,
                        name=s.name,
                        schedule_days=s.schedule_days,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        student_count=len(s.enrollments),
                    )
                    for s in cls.sections
                ],
                already_enrolled_section_id=my_enrollments.get(cls.id),
                pending_request_section_id=my_pending.get(cls.id),
            )
        )
    return out


@router.get("/all", response_model=list[ClassOut])
def all_classes(
    university_id: int | None = None,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(UserRole.admin)),
):
    """Admin oversight view — every class across every teacher and university,
    optionally filtered to one university."""
    query = db.query(Class).options(
        joinedload(Class.university),
        joinedload(Class.sections).selectinload(Section.enrollments),
    )
    if university_id:
        query = query.filter(Class.university_id == university_id)
    return [_class_to_out(c) for c in query.all()]


@router.post("/{class_id}/reassign-teacher", response_model=ClassOut)
def reassign_teacher(
    class_id: int,
    new_teacher_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(UserRole.admin)),
):
    """Admin override: move a class to a different teacher. Use sparingly —
    this doesn't touch existing enrollments or sections, only ownership."""
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found.")

    new_teacher = db.query(User).filter(User.id == new_teacher_id, User.role == UserRole.teacher).first()
    if not new_teacher:
        raise HTTPException(status_code=404, detail="Target teacher not found.")

    cls.teacher_id = new_teacher_id
    db.commit()
    return _class_to_out(load_class_for_out(db, cls.id))


@router.get("/{class_id}/roster")
def class_roster(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Enrolled students for a class, grouped by section. A teacher may only
    view the roster for their own class; a student gets a 403 (rosters aren't
    a student-facing view — Browse Classes shows sections/counts, not names)."""
    cls = db.query(Class).options(joinedload(Class.sections)).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found.")
    if current_user.role == UserRole.teacher and cls.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't have access to this class.")
    if current_user.role == UserRole.student:
        raise HTTPException(status_code=403, detail="Not available to students.")

    enrollments = (
        db.query(Enrollment)
        .options(joinedload(Enrollment.student))
        .filter(Enrollment.class_id == class_id)
        .all()
    )
    enrollments_by_section: dict[int, list[Enrollment]] = {}
    for enrollment in enrollments:
        enrollments_by_section.setdefault(enrollment.section_id, []).append(enrollment)

    result = []
    for section in cls.sections:
        section_enrollments = enrollments_by_section.get(section.id, [])
        result.append(
            {
                "section_id": section.id,
                "section_name": section.name,
                "students": [
                    {
                        "id": e.student.id,
                        "name": e.student.name,
                        "email": e.student.email,
                        "avatar_url": e.student.avatar_url,
                    }
                    for e in section_enrollments
                ],
            }
        )
    return result


@router.get("/sections/{section_id}/roster")
def section_roster(
    section_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return only the students enrolled in one section. Attendance screens
    should use this endpoint instead of loading the entire class roster."""
    section = (
        db.query(Section)
        .options(joinedload(Section.class_))
        .filter(Section.id == section_id)
        .first()
    )
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")
    if current_user.role == UserRole.teacher and section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't have access to this section.")
    if current_user.role == UserRole.student:
        raise HTTPException(status_code=403, detail="Not available to students.")

    enrollments = (
        db.query(Enrollment)
        .options(joinedload(Enrollment.student))
        .filter(Enrollment.section_id == section.id)
        .all()
    )
    return {
        "section_id": section.id,
        "section_name": section.name,
        "students": [
            {
                "id": e.student.id,
                "name": e.student.name,
                "email": e.student.email,
                "avatar_url": e.student.avatar_url,
            }
            for e in enrollments
        ],
    }


@router.get("/{class_id}", response_model=ClassOut)
def get_class(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cls = load_class_for_out(db, class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found.")

    # Ownership check: a teacher may only view their own class; admin can view any;
    # a student may only view a class they're enrolled in or have requested (kept
    # permissive here — the browse endpoint is the primary discovery path).
    if current_user.role == UserRole.teacher and cls.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't have access to this class.")

    return _class_to_out(cls)


@router.patch("/{class_id}", response_model=ClassOut)
def update_class(
    class_id: int,
    payload: ClassUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found.")
    if current_user.role == UserRole.teacher and cls.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this class.")
    elif current_user.role == UserRole.student:
        raise HTTPException(status_code=403, detail="Students can't edit classes.")

    if payload.name is not None:
        cls.name = payload.name
    if payload.code is not None:
        existing_code = db.query(Class).filter(Class.code == payload.code, Class.id != class_id).first()
        if existing_code:
            raise HTTPException(status_code=400, detail=f"Class code '{payload.code}' is already in use.")
        cls.code = payload.code
    if payload.subject is not None:
        cls.subject = payload.subject
    if payload.university_id is not None:
        from app.models.university import University
        university = db.query(University).filter(University.id == payload.university_id).first()
        if not university:
            raise HTTPException(status_code=404, detail="University not found.")
        cls.university_id = payload.university_id

    if payload.sections:
        section_ids = [section_update.id for section_update in payload.sections]
        sections_by_id = {
            section.id: section
            for section in db.query(Section)
            .filter(Section.class_id == class_id, Section.id.in_(section_ids))
            .all()
        }
        for section_update in payload.sections:
            section = sections_by_id.get(section_update.id)
            if not section:
                continue
            if section_update.name is not None:
                section.name = section_update.name.strip()
            if section_update.schedule_days is not None:
                section.schedule_days = section_update.schedule_days
            if section_update.start_time is not None:
                section.start_time = section_update.start_time
            if section_update.end_time is not None:
                section.end_time = section_update.end_time

    _validate_schedule(
        db,
        cls.teacher_id,
        [(s.schedule_days, s.start_time, s.end_time) for s in cls.sections],
        exclude_class_id=cls.id,
    )

    db.commit()
    return _class_to_out(load_class_for_out(db, cls.id))


@router.delete("/{class_id}", status_code=204)
def delete_class(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found.")
    if current_user.role == UserRole.teacher and cls.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this class.")
    elif current_user.role == UserRole.student:
        raise HTTPException(status_code=403, detail="Students can't delete classes.")

    # Sections/Enrollments/AttendanceSessions cascade via their relationships,
    # but EnrollmentRequest and LeaveRequest don't have a defined cascade
    # relationship — clean those up explicitly or this hits a foreign-key
    # violation on a real MySQL database (SQLite silently allows it, which
    # would have hidden this bug in testing).
    db.query(EnrollmentRequest).filter(EnrollmentRequest.class_id == class_id).delete()
    from app.models.leave import LeaveRequest
    db.query(LeaveRequest).filter(LeaveRequest.class_id == class_id).delete()

    db.delete(cls)
    db.commit()
    return None

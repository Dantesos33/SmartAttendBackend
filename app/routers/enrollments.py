from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
import os
import shutil
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Section
from app.models.enrollment import Enrollment, EnrollmentRequest, EnrollmentRequestStatus
from app.schemas.enrollment import EnrollmentRequestCreate, EnrollmentRequestOut, EnrollmentOut
from app.schemas.class_ import AddStudentsBulkRequest, AddStudentResult
from app.core.deps import get_current_user, require_role
from app.core.notifications import notify
from app.core.security import hash_password
from app.core.db_helpers import get_section_with_class
from app.models.notification import NotificationType

router = APIRouter(prefix="/enrollments", tags=["enrollments"])


@router.post("/request", response_model=EnrollmentRequestOut, status_code=201)
def request_enrollment(
    payload: EnrollmentRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.student)),
):
    section = get_section_with_class(db, payload.section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")

    class_id = section.class_id

    # Already enrolled in *some* section of this class? Block — this is the
    # "can't enroll in a different section of a class you're already in" rule.
    already_enrolled = (
        db.query(Enrollment)
        .filter(Enrollment.student_id == current_user.id, Enrollment.class_id == class_id)
        .first()
    )
    if already_enrolled:
        raise HTTPException(
            status_code=409,
            detail="You're already enrolled in a section of this class.",
        )

    # Already has a pending request for this class (possibly a different section)?
    existing_pending = (
        db.query(EnrollmentRequest)
        .filter(
            EnrollmentRequest.student_id == current_user.id,
            EnrollmentRequest.class_id == class_id,
            EnrollmentRequest.status == EnrollmentRequestStatus.pending,
        )
        .first()
    )
    if existing_pending:
        raise HTTPException(
            status_code=409,
            detail="You already have a pending request for this class.",
        )

    req = EnrollmentRequest(student_id=current_user.id, section_id=section.id, class_id=class_id)
    db.add(req)
    db.flush()

    # Notify the owning teacher of the new request.
    teacher_id = section.class_.teacher_id
    notify(
        db,
        user_id=teacher_id,
        type_=NotificationType.enrollment_request_received,
        title=f"New enrollment request for {section.class_.name} ({section.name})",
        body=f"{current_user.name} requested to join.",
        related_id=req.id,
    )

    db.commit()
    db.refresh(req)
    return EnrollmentRequestOut(
        id=req.id,
        student_id=req.student_id,
        student_name=current_user.name,
        section_id=req.section_id,
        section_name=section.name,
        class_id=req.class_id,
        class_name=section.class_.name,
        status=req.status,
        requested_at=req.requested_at,
    )


@router.get("/requests/pending", response_model=list[EnrollmentRequestOut])
def pending_requests_for_my_classes(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """Only requests for classes this teacher owns — scoped by ownership, same
    rule as the classes list itself."""
    requests = (
        db.query(EnrollmentRequest)
        .join(Section, EnrollmentRequest.section_id == Section.id)
        .options(
            joinedload(EnrollmentRequest.student),
            joinedload(EnrollmentRequest.section).joinedload(Section.class_),
        )
        .filter(
            Section.class_.has(teacher_id=current_user.id),
            EnrollmentRequest.status == EnrollmentRequestStatus.pending,
        )
        .all()
    )
    return [
        EnrollmentRequestOut(
            id=r.id,
            student_id=r.student_id,
            student_name=r.student.name,
            section_id=r.section_id,
            section_name=r.section.name,
            class_id=r.class_id,
            class_name=r.section.class_.name,
            status=r.status,
            requested_at=r.requested_at,
        )
        for r in requests
    ]


def _get_owned_request(db: Session, request_id: int, teacher: User) -> EnrollmentRequest:
    req = (
        db.query(EnrollmentRequest)
        .options(joinedload(EnrollmentRequest.section).joinedload(Section.class_))
        .filter(EnrollmentRequest.id == request_id)
        .first()
    )
    if not req:
        raise HTTPException(status_code=404, detail="Enrollment request not found.")
    if req.section.class_.teacher_id != teacher.id:
        raise HTTPException(status_code=403, detail="This request isn't for one of your classes.")
    if req.status != EnrollmentRequestStatus.pending:
        raise HTTPException(status_code=409, detail=f"Request already {req.status.value}.")
    return req


@router.post("/requests/{request_id}/accept", response_model=EnrollmentOut)
def accept_request(
    request_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    req = _get_owned_request(db, request_id, current_user)

    # Double-check the uniqueness rule hasn't been violated by another accepted
    # request in the meantime (e.g. two pending requests for different sections
    # of the same class, both about to be accepted).
    already_enrolled = (
        db.query(Enrollment)
        .filter(Enrollment.student_id == req.student_id, Enrollment.class_id == req.class_id)
        .first()
    )
    if already_enrolled:
        raise HTTPException(
            status_code=409,
            detail="Student is already enrolled in a section of this class.",
        )

    enrollment = Enrollment(student_id=req.student_id, section_id=req.section_id, class_id=req.class_id)
    db.add(enrollment)

    req.status = EnrollmentRequestStatus.accepted
    from datetime import datetime

    req.resolved_at = datetime.utcnow()

    notify(
        db,
        user_id=req.student_id,
        type_=NotificationType.enrollment_accepted,
        title=f"You've been enrolled in {req.section.class_.name}",
        body=f"Section {req.section.name} — you're all set.",
        related_id=req.class_id,
    )

    db.commit()
    db.refresh(enrollment)
    return enrollment


@router.post("/requests/{request_id}/reject", status_code=204)
def reject_request(
    request_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    req = _get_owned_request(db, request_id, current_user)
    req.status = EnrollmentRequestStatus.rejected
    from datetime import datetime

    req.resolved_at = datetime.utcnow()

    notify(
        db,
        user_id=req.student_id,
        type_=NotificationType.enrollment_rejected,
        title=f"Your request to join {req.section.class_.name} was declined",
        related_id=req.class_id,
    )

    db.commit()
    return None


@router.get("/mine", response_model=list[EnrollmentOut])
def my_enrollments(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.student)),
):
    return db.query(Enrollment).filter(Enrollment.student_id == current_user.id).all()


@router.post("/admin/enroll", response_model=EnrollmentOut, status_code=201)
def admin_direct_enroll(
    student_id: int,
    section_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(UserRole.admin)),
):
    """Admin override: enroll a student directly, bypassing the request/approval
    flow. Still enforces the one-section-per-class rule, and still notifies the
    student — an admin action is never silent."""
    student = db.query(User).filter(User.id == student_id, User.role == UserRole.student).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    section = get_section_with_class(db, section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")

    already_enrolled = (
        db.query(Enrollment)
        .filter(Enrollment.student_id == student_id, Enrollment.class_id == section.class_id)
        .first()
    )
    if already_enrolled:
        raise HTTPException(
            status_code=409, detail="Student is already enrolled in a section of this class."
        )

    enrollment = Enrollment(student_id=student_id, section_id=section_id, class_id=section.class_id)
    db.add(enrollment)

    notify(
        db,
        user_id=student_id,
        type_=NotificationType.enrollment_accepted,
        title=f"You've been enrolled in {section.class_.name}",
        body=f"Section {section.name} — added by an administrator.",
        related_id=section.class_id,
    )

    db.commit()
    db.refresh(enrollment)
    return enrollment


@router.delete("/admin/unenroll", status_code=204)
def admin_direct_unenroll(
    student_id: int,
    class_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(UserRole.admin)),
):
    """Admin override: remove a student's enrollment directly. Still notifies
    the student."""
    enrollment = (
        db.query(Enrollment)
        .options(joinedload(Enrollment.section).joinedload(Section.class_))
        .filter(Enrollment.student_id == student_id, Enrollment.class_id == class_id)
        .first()
    )
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found.")

    class_name = enrollment.section.class_.name
    db.delete(enrollment)

    notify(
        db,
        user_id=student_id,
        type_=NotificationType.enrollment_rejected,
        title=f"You've been unenrolled from {class_name}",
        body="This was done by an administrator.",
        related_id=class_id,
    )

    db.commit()
    return None


@router.post("/add-students", response_model=list[AddStudentResult])
def teacher_add_students(
    payload: AddStudentsBulkRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """Teacher adds one or more students directly to a section — one-by-one
    is just a list of length 1. For a genuinely new email, creates a real
    account with a random default password (must_change_password=True,
    forces a change on first login) and enrolls them immediately — no
    request/approval needed, the teacher already chose them. For an email
    that already has an account, enrolls the existing student instead of
    creating a duplicate. Either way, the student gets the same
    'enrolled_by_teacher' notification, and the one-section-per-class rule
    still applies."""
    section = get_section_with_class(db, payload.section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")
    if section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")

    student_ids = [entry.student_id.strip() for entry in payload.students if entry.student_id.strip()]
    existing_users = {
        user.student_id: user
        for user in db.query(User).filter(User.student_id.in_(student_ids)).all()
    } if student_ids else {}
    existing_user_ids = [user.id for user in existing_users.values()]
    enrolled_user_ids = {
        row.student_id
        for row in db.query(Enrollment.student_id)
        .filter(
            Enrollment.class_id == section.class_id,
            Enrollment.student_id.in_(existing_user_ids),
        )
        .all()
    } if existing_user_ids else set()

    results: list[AddStudentResult] = []

    for entry in payload.students:
        sid = entry.student_id.strip()
        existing = existing_users.get(sid)

        if existing and existing.role != UserRole.student:
            results.append(AddStudentResult(student_id=sid, status="error", message="This ID belongs to a non-student account."))
            continue

        if existing:
            if existing.id in enrolled_user_ids:
                results.append(AddStudentResult(student_id=sid, status="error", message="Already enrolled in a section of this class."))
                continue
            student = existing
            status_label = "existing_enrolled"
            enrolled_user_ids.add(existing.id)
            message = f"{existing.name} already has an account and was enrolled in this class."
        else:
            # Default password is the student's own ID + "@123" — simple and
            # memorable for a first login, forced to change immediately after
            # via must_change_password.
            temp_password = f"{sid}@123"
            student = User(
                name=entry.name.strip(),
                student_id=sid,
                password_hash=hash_password(temp_password),
                role=UserRole.student,
                must_change_password=True,
            )
            db.add(student)
            db.flush()
            status_label = "created_and_enrolled"
            existing_users[sid] = student
            enrolled_user_ids.add(student.id)

        enrollment = Enrollment(student_id=student.id, section_id=section.id, class_id=section.class_id)
        db.add(enrollment)

        notify(
            db,
            user_id=student.id,
            type_=NotificationType.enrolled_by_teacher,
            title=f"You've been enrolled in {section.class_.name} by {current_user.name}",
            body=f"Section {section.name}."
            + (f" Log in with ID {sid} and password {sid}@123 — you'll be asked to change it and verify your profile." if status_label == "created_and_enrolled" else ""),
            related_id=section.class_id,
        )

        message = (
            f"Account created (login ID: {sid}, temp password: {sid}@123) and enrolled."
            if status_label == "created_and_enrolled"
            else message
        )
        results.append(
            AddStudentResult(
                student_id=sid,
                user_id=student.id,
                status=status_label,
                message=message,
            )
        )

    db.commit()
    return results


@router.post("/add-student-with-photo", response_model=AddStudentResult)
async def teacher_add_student_with_photo(
    section_id: int = Form(...),
    name: str = Form(...),
    student_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher)),
):
    """Add an unrecognized face from attendance review — enrolls the student,
    registers the cropped face for AI recognition, and sets their profile photo
    so they skip the verify-profile gate on first login."""
    from app.routers.recognition import attendance_system

    section = get_section_with_class(db, section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")
    if section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")

    sid = student_id.strip()
    display_name = name.strip()
    if not sid or not display_name:
        raise HTTPException(status_code=400, detail="Name and student ID are required.")

    existing = db.query(User).filter(User.student_id == sid).first()
    if existing and existing.role != UserRole.student:
        return AddStudentResult(
            student_id=sid,
            status="error",
            message="This ID belongs to a non-student account.",
        )

    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/enroll_face_{sid}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if existing:
            already_enrolled = (
                db.query(Enrollment)
                .filter(
                    Enrollment.student_id == existing.id,
                    Enrollment.class_id == section.class_id,
                )
                .first()
            )
            if already_enrolled:
                return AddStudentResult(
                    student_id=sid,
                    user_id=existing.id,
                    status="error",
                    message="Already enrolled in a section of this class.",
                )

            enrollment = Enrollment(
                student_id=existing.id,
                section_id=section.id,
                class_id=section.class_id,
            )
            db.add(enrollment)

            notify(
                db,
                user_id=existing.id,
                type_=NotificationType.enrolled_by_teacher,
                title=f"You've been enrolled in {section.class_.name} by {current_user.name}",
                body=f"Section {section.name}.",
                related_id=section.class_id,
            )

            db.commit()
            return AddStudentResult(
                student_id=sid,
                user_id=existing.id,
                status="existing_enrolled",
                message=f"{existing.name} already has an account and was enrolled in this class.",
            )

        is_valid, message = attendance_system.verify_face_quality(temp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        temp_password = f"{sid}@123"
        student = User(
            name=display_name,
            student_id=sid,
            password_hash=hash_password(temp_password),
            role=UserRole.student,
            must_change_password=True,
        )
        db.add(student)
        db.flush()

        success, reg_message, encoding = attendance_system.register_student_face(
            temp_path, student.id, student.name, roll=sid
        )
        if not success:
            raise HTTPException(status_code=400, detail=reg_message)

        if encoding is not None:
            student.face_encoding_json = attendance_system._encoding_to_json(encoding)

        student.avatar_url = f"/media/known_students/{student.id}.jpg"

        enrollment = Enrollment(
            student_id=student.id,
            section_id=section.id,
            class_id=section.class_id,
        )
        db.add(enrollment)

        notify(
            db,
            user_id=student.id,
            type_=NotificationType.enrolled_by_teacher,
            title=f"You've been enrolled in {section.class_.name} by {current_user.name}",
            body=(
                f"Section {section.name}. Your profile photo was added from attendance capture. "
                f"Log in with ID {sid} and password {sid}@123."
            ),
            related_id=section.class_id,
        )

        db.commit()
        return AddStudentResult(
            student_id=sid,
            user_id=student.id,
            status="created_and_enrolled",
            message=f"Account created (login ID: {sid}, temp password: {sid}@123) and enrolled.",
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

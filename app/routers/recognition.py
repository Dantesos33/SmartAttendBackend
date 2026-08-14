import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from app.recognition_engine import ClassroomAttendanceSystem
from app.core.deps import require_role, get_db
from app.core.face_encoding_sync import (
    sync_student_encodings,
    students_missing_face_data,
)
from app.models.user import User, UserRole
from app.models.enrollment import Enrollment
from app.core.db_helpers import get_section_with_class

router = APIRouter(tags=["recognition"])

attendance_system = ClassroomAttendanceSystem(known_students_dir="known_students")


@router.get("/students")
def get_students(_: User = Depends(require_role(UserRole.teacher, UserRole.admin))):
    """Every face the recognition engine has been trained on."""
    return {"status": "success", "students": attendance_system.get_all_students_stats()}


@router.get("/students/{name}/history")
def get_student_history(name: str, _: User = Depends(require_role(UserRole.teacher, UserRole.admin))):
    stats = attendance_system.get_student_stats(name)
    return {"status": "success", "data": stats}


@router.delete("/students/{name}")
def delete_student(name: str, _: User = Depends(require_role(UserRole.teacher, UserRole.admin))):
    success, message = attendance_system.remove_student(name)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return {"status": "success", "message": message}


@router.post("/sync-encodings")
def sync_face_encodings(
    db: Session = Depends(get_db),
    _: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    """Rebuild missing face embeddings from on-disk profile photos."""
    synced = sync_student_encodings(db, attendance_system)
    return {
        "status": "success",
        "message": f"Synced {synced} student face embedding(s) from profile photos.",
        "synced_count": synced,
    }


@router.post("/register")
async def register_student(
    name: str = Form(...),
    roll: str = Form(...),
    file: UploadFile = File(...),
    _: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    try:
        os.makedirs("known_students", exist_ok=True)
        file_ext = os.path.splitext(file.filename)[1]
        safe_name = name.replace(" ", "_")
        file_path = f"known_students/{safe_name}{file_ext}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        success, message = attendance_system.load_student_image(file_path, name, roll=roll)

        if success:
            return {"status": "success", "message": message, "student": name}
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize")
async def recognize_classroom(
    file: UploadFile = File(...),
    section_id: int | None = Form(None),
    tolerance: float = Form(0.65),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    """Raw AI output only — this does NOT save attendance."""
    try:
        allowed_student_ids = None
        if section_id is not None:
            section = get_section_with_class(db, section_id)
            if not section:
                raise HTTPException(status_code=404, detail="Section not found.")
            if (
                current_user.role == UserRole.teacher
                and section.class_.teacher_id != current_user.id
            ):
                raise HTTPException(status_code=403, detail="You don't own this section's class.")
            allowed_student_ids = [
                row.student_id
                for row in db.query(Enrollment.student_id)
                .filter(Enrollment.section_id == section_id)
                .all()
            ]

        # Students may have avatar_url from before encodings were stored in the DB.
        # Rebuild embeddings from any profile photos still on disk.
        if allowed_student_ids is not None:
            sync_student_encodings(db, attendance_system, allowed_student_ids)
        else:
            sync_student_encodings(db, attendance_system)

        encoding_query = db.query(User.id, User.name, User.face_encoding_json).filter(
            User.face_encoding_json.isnot(None),
            User.role == UserRole.student,
        )
        if allowed_student_ids is not None:
            if allowed_student_ids:
                encoding_query = encoding_query.filter(User.id.in_(allowed_student_ids))
            else:
                encoding_query = encoding_query.filter(User.id == -1)
        db_encoding_rows = encoding_query.all()

        os.makedirs("temp", exist_ok=True)
        # Unique temporary names prevent repeated captures from reusing an old
        # image when the frontend sends the same filename again.
        suffix = os.path.splitext(file.filename or "capture.jpg")[1].lower() or ".jpg"
        temp_path = os.path.join("temp", f"attendance_{uuid.uuid4().hex}{suffix}")

        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            attendance_data, message = attendance_system.recognize_classroom(
                temp_path,
                tolerance=tolerance,
                allowed_student_ids=allowed_student_ids,
                db_encoding_rows=db_encoding_rows,
            )

            if not attendance_data:
                raise HTTPException(status_code=400, detail=message)

            if allowed_student_ids is not None:
                missing = students_missing_face_data(db, allowed_student_ids)
                attendance_data["students_missing_face_data"] = missing
                if missing and not attendance_data.get("recognition_available"):
                    names = ", ".join(item["name"] for item in missing[:5])
                    extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
                    attendance_data["warning_message"] = (
                        "These enrolled students need to re-upload their profile photo "
                        f"so face recognition can work: {names}{extra}."
                    )

            return {"status": "success", "message": message, "data": attendance_data}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

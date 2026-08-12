import os
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from app.recognition_engine import ClassroomAttendanceSystem
from app.core.deps import require_role, get_db
from app.models.user import User, UserRole
from app.models.enrollment import Enrollment
from app.core.db_helpers import get_section_with_class

router = APIRouter(tags=["recognition"])

# Single shared instance, loaded once at import time — same lifecycle as the
# standalone recognition service had, just living inside this process now.
attendance_system = ClassroomAttendanceSystem(known_students_dir="known_students")


@router.get("/students")
def get_students(_: User = Depends(require_role(UserRole.teacher, UserRole.admin))):
    """Every face the recognition engine has been trained on. Note this is
    NOT the same thing as a class roster (see GET /classes/{id}/roster) — this
    is a flat list scoped to the recognition engine's own storage, independent
    of which class/section a student is actually enrolled in."""
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
    tolerance: float = Form(0.45),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    """Raw AI output only — this does NOT save attendance. When section_id is
    provided, only students enrolled in that section can be matched."""
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

        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/{file.filename}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        attendance_data, message = attendance_system.recognize_classroom(
            temp_path,
            tolerance=tolerance,
            allowed_student_ids=allowed_student_ids,
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not attendance_data:
            raise HTTPException(status_code=400, detail=message)

        return {"status": "success", "message": message, "data": attendance_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

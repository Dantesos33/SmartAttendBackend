import os
import shutil
import json
import uuid
from datetime import datetime
import face_recognition
import cv2
import numpy as np

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
RECOGNITION_JOB_DIR = "temp/recognition_jobs"
os.makedirs(RECOGNITION_JOB_DIR, exist_ok=True)

def _job_paths(job_id: str):
    base = os.path.join(RECOGNITION_JOB_DIR, job_id)
    return base, base + ".json"

def _save_job(job_id: str, job: dict):
    base, meta = _job_paths(job_id)
    os.makedirs(base, exist_ok=True)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(job, f)

def _load_job(job_id: str):
    base, meta = _job_paths(job_id)
    if not os.path.exists(meta):
        raise HTTPException(status_code=404, detail="Recognition job not found or expired.")
    with open(meta, "r", encoding="utf-8") as f:
        job = json.load(f)
    if not os.path.exists(job.get("image_path", "")):
        raise HTTPException(status_code=404, detail="Recognition image is no longer available.")
    return job

def _cleanup_job(job_id: str):
    base, meta = _job_paths(job_id)
    shutil.rmtree(base, ignore_errors=True)
    try:
        os.remove(meta)
    except FileNotFoundError:
        pass

def _validate_section(db, section_id, current_user):
    if section_id is None:
        raise HTTPException(status_code=400, detail="section_id is required for attendance recognition.")
    section = get_section_with_class(db, section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found.")
    if current_user.role == UserRole.teacher and section.class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't own this section's class.")
    return section



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


@router.post("/recognize/detect")
async def detect_faces_stage(
    file: UploadFile = File(...),
    section_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    """Stage 1: detect every face. No identity or enrollment decision is made here."""
    _validate_section(db, section_id, current_user)
    job_id = uuid.uuid4().hex
    base, _ = _job_paths(job_id)
    os.makedirs(base, exist_ok=True)
    image_path = os.path.join(base, "original.jpg")
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        image = face_recognition.load_image_file(image_path)
        locations = attendance_system._detect_face_locations(image)
        faces = [
            {"face_index": i, "location": {"top": int(t), "right": int(r), "bottom": int(b), "left": int(l)}}
            for i, (t, r, b, l) in enumerate(locations)
        ]
        job = {
            "job_id": job_id, "section_id": section_id, "image_path": image_path,
            "face_locations": [list(x) for x in locations], "faces": faces,
            "stage": "faces_detected", "created_at": datetime.utcnow().isoformat(),
        }
        _save_job(job_id, job)
        return {"status": "success", "job_id": job_id, "stage": "detecting_faces", "data": {"faces_detected": len(faces), "faces": faces}}
    except Exception:
        _cleanup_job(job_id)
        raise

@router.post("/recognize/check-faces")
def check_faces_stage(
    job_id: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    """Stage 2: classify detected faces as masked/clear and prepare clear-face embeddings."""
    job = _load_job(job_id)
    section = _validate_section(db, job["section_id"], current_user)
    image = face_recognition.load_image_file(job["image_path"])
    rgb = np.ascontiguousarray(image)
    face_locations = [tuple(x) for x in job["face_locations"]]
    results = []
    for i, loc in enumerate(face_locations):
        top, right, bottom, left = loc
        encoding = None
        try:
            encs = face_recognition.face_encodings(rgb, known_face_locations=[loc], num_jitters=1)
            if encs:
                encoding = [float(x) for x in encs[0]]
        except Exception as exc:
            print(f"Face encoding failed for face {i}: {exc}")

        # Detection is already complete. A face that cannot be encoded is kept
        # as masked/occluded rather than being dropped from attendance.
        status = attendance_system.classify_face_occlusion(
            rgb, loc, encoding_available=encoding is not None
        )
        crop = attendance_system._encode_rgb_crop_base64(rgb, top, right, bottom, left, quality=96, padded=True)
        results.append({
            "face_index": i, "face_status": status,
            "location": {"top": top, "right": right, "bottom": bottom, "left": left},
            "encoding": encoding, "crop_base64": crop,
        })
    job["faces"] = results
    job["stage"] = "faces_checked"
    _save_job(job_id, job)
    return {"status": "success", "job_id": job_id, "stage": "checking_faces", "data": {"faces": [{k:v for k,v in f.items() if k != "encoding"} for f in results]}}

@router.post("/recognize/check-enrollment")
def check_enrollment_stage(
    job_id: str = Form(...),
    tolerance: float = Form(0.58),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    """Stage 3/4: match clear faces ONLY against this section, then calculate present/absent."""
    job = _load_job(job_id)
    section = _validate_section(db, job["section_id"], current_user)
    allowed_student_ids = [row.student_id for row in db.query(Enrollment.student_id).filter(Enrollment.section_id == section.id).all()]
    sync_student_encodings(db, attendance_system, allowed_student_ids)
    rows = db.query(User.id, User.name, User.face_encoding_json).filter(User.id.in_(allowed_student_ids), User.role == UserRole.student).all() if allowed_student_ids else []
    attendance_system.prepare_known_faces(db_encoding_rows=rows, allowed_student_ids=allowed_student_ids)
    allowed_set = set(allowed_student_ids)

    # Match all clear faces globally, not one face at a time. This prevents the
    # same enrolled student from being assigned to multiple detected faces and
    # prevents a loose threshold from marking unrelated people present.
    clear_encodings = {
        int(face["face_index"]): np.array(face["encoding"], dtype=np.float64)
        for face in job["faces"]
        if face["face_status"] == "clear" and face.get("encoding")
    }
    assignments = attendance_system._assign_faces_to_students(
        clear_encodings,
        tolerance=float(tolerance),
        allowed_set=allowed_set,
        min_confidence=0.42,
    )

    present_ids = []
    final_faces = []
    for face in job["faces"]:
        face_index = int(face["face_index"])
        student_id = None
        name = "Unrecognized"
        confidence = 0.0
        status = face["face_status"]

        if status == "masked":
            name = "Masked"
        elif face_index in assignments:
            student_id, name, confidence = assignments[face_index]
            if student_id is None:
                name = "Unrecognized"
            elif student_id not in present_ids:
                present_ids.append(student_id)

        final_faces.append({
            "face_index": face_index, "student_id": student_id, "name": name,
            "confidence": float(confidence), "face_status": status,
            "attendance_status": "masked" if status == "masked" else ("present" if student_id else "unrecognized"),
            "location": face["location"], "crop_base64": face.get("crop_base64"),
        })
    absent_ids = [sid for sid in allowed_student_ids if sid not in present_ids]
    image = face_recognition.load_image_file(job["image_path"])
    annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face in final_faces:
        loc = face["location"]; top,right,bottom,left = loc["top"],loc["right"],loc["bottom"],loc["left"]
        status = face["attendance_status"]
        color = (0,255,0) if status == "present" else ((0,165,255) if status == "masked" else (0,0,255))
        label = face["name"] if status != "present" else f"{face['name']} ({face['confidence']:.0%})"
        cv2.rectangle(annotated, (left,top), (right,bottom), color, 3)
        y=max(top-8,20); (tw,th),_=cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,.55,1)
        cv2.rectangle(annotated,(left,y-th-4),(left+tw+6,y+4),color,cv2.FILLED)
        cv2.putText(annotated,label,(left+3,y-2),cv2.FONT_HERSHEY_DUPLEX,.55,(255,255,255),1)
    annotated_b64 = attendance_system._encode_bgr_jpeg_base64(annotated, quality=88, max_edge=1200)
    missing = students_missing_face_data(db, allowed_student_ids)
    result = {
        "section_id": section.id, "faces_detected": len(final_faces), "present_student_ids": present_ids,
        "absent_student_ids": absent_ids, "present_count": len(present_ids), "absent_count": len(absent_ids),
        "masked_count": sum(1 for f in final_faces if f["attendance_status"] == "masked"),
        "unknown_faces": sum(1 for f in final_faces if f["attendance_status"] == "unrecognized"),
        "face_details": final_faces, "annotated_image_base64": annotated_b64,
        "enrolled_with_face_photos": len(attendance_system.known_face_ids), "recognition_available": bool(attendance_system.known_face_encodings),
        "students_missing_face_data": missing,
    }
    if missing and not result["recognition_available"]:
        names = ", ".join(item["name"] for item in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        result["warning_message"] = (
            "These enrolled students need to re-upload their profile photo "
            f"so face recognition can work: {names}{extra}."
        )
    job["stage"] = "complete"; job["result"] = result; _save_job(job_id, job)
    response = {"status":"success", "job_id":job_id, "stage":"complete", "data":result}
    _cleanup_job(job_id)
    return response

@router.post("/recognize")
async def recognize_classroom(
    file: UploadFile = File(...),
    section_id: int = Form(...),
    tolerance: float = Form(0.55),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.teacher, UserRole.admin)),
):
    """Raw AI output only — this does NOT save attendance."""
    try:
        section = _validate_section(db, section_id, current_user)
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
        temp_path = f"temp/{file.filename}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        attendance_data, message = attendance_system.recognize_classroom(
            temp_path,
            tolerance=tolerance,
            allowed_student_ids=allowed_student_ids,
            db_encoding_rows=db_encoding_rows,
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)

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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

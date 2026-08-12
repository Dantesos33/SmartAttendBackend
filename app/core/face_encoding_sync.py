"""Keep face embeddings in the DB in sync with profile photos on disk."""

import os

from sqlalchemy.orm import Session

from app.models.user import User, UserRole
from app.recognition_engine import ClassroomAttendanceSystem


def resolve_student_photo_path(student_id: int, avatar_url: str | None = None) -> str | None:
    candidates: list[str] = []
    if avatar_url:
        avatar = avatar_url.split("?", 1)[0]
        if "/known_students/" in avatar:
            filename = avatar.rsplit("/", 1)[-1]
            candidates.append(os.path.join("known_students", filename))
    for ext in (".jpg", ".jpeg", ".png"):
        candidates.append(os.path.join("known_students", f"{student_id}{ext}"))

    seen: set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    return None


def ensure_student_encoding(
    db: Session,
    attendance_system: ClassroomAttendanceSystem,
    user: User,
    *,
    commit: bool = False,
) -> bool:
    """Build and persist a face embedding when the photo file exists on disk."""
    if user.face_encoding_json:
        return True

    photo_path = resolve_student_photo_path(user.id, user.avatar_url)
    if not photo_path:
        return False

    success, _, encoding = attendance_system._register_encoding(
        photo_path, user.id, user.name
    )
    if not success or encoding is None:
        return False

    user.face_encoding_json = attendance_system._encoding_to_json(encoding)
    if commit:
        db.commit()
    return True


def sync_student_encodings(
    db: Session,
    attendance_system: ClassroomAttendanceSystem,
    student_ids: list[int] | None = None,
) -> int:
    """Backfill missing DB embeddings from on-disk profile photos."""
    query = db.query(User).filter(User.role == UserRole.student)
    if student_ids is not None:
        if not student_ids:
            return 0
        query = query.filter(User.id.in_(student_ids))

    synced = 0
    for user in query.all():
        if user.face_encoding_json:
            continue
        if ensure_student_encoding(db, attendance_system, user):
            synced += 1

    if synced:
        db.commit()
    return synced


def students_missing_face_data(
    db: Session,
    student_ids: list[int],
) -> list[dict]:
    """Enrolled students who appear to have a profile but no usable face data."""
    missing: list[dict] = []
    for sid in student_ids:
        user = db.query(User).filter(User.id == sid).first()
        if not user:
            continue
        has_encoding = bool(user.face_encoding_json)
        has_photo = resolve_student_photo_path(user.id, user.avatar_url) is not None
        if has_encoding or has_photo:
            continue
        missing.append(
            {
                "student_id": user.id,
                "name": user.name,
                "has_avatar_url": bool(user.avatar_url),
            }
        )
    return missing

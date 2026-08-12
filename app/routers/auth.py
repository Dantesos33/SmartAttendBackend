import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserRole
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserOut,
    ChangePasswordRequest,
    ForgotPasswordVerifyRequest,
    ForgotPasswordResetRequest,
    UpdateProfileRequest,
)
from app.core.security import hash_password, verify_password, create_access_token
from app.core.deps import get_current_user, require_role

router = APIRouter(prefix="/auth", tags=["auth"])

_RESET_MISMATCH = "No account matches these details."


def _find_user_for_password_reset(
    role: str,
    email: str | None,
    student_id: str | None,
    db: Session,
) -> User:
    """In-app reset: students by student ID, teachers/admins by email."""
    if role == "student":
        sid = (student_id or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail=_RESET_MISMATCH)
        user = (
            db.query(User)
            .filter(User.student_id == sid, User.role == UserRole.student)
            .first()
        )
    else:
        if not email:
            raise HTTPException(status_code=400, detail=_RESET_MISMATCH)
        user = (
            db.query(User)
            .filter(
                User.email == email,
                User.role.in_([UserRole.teacher, UserRole.admin]),
            )
            .first()
        )

    if not user:
        raise HTTPException(status_code=400, detail=_RESET_MISMATCH)
    return user


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if payload.role == UserRole.admin:
        raise HTTPException(
            status_code=403,
            detail="Admin accounts cannot be self-registered. Contact a system administrator.",
        )

    if payload.email:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            # Same generic message as a failed login would eventually see, so this
            # endpoint doesn't become a way to enumerate which emails/IDs are registered.
            raise HTTPException(status_code=400, detail="Unable to register with these details.")
    if payload.student_id:
        existing = db.query(User).filter(User.student_id == payload.student_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Unable to register with these details.")

    user = User(
        name=payload.name,
        email=payload.email,
        student_id=payload.student_id,
        password_hash=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    # identifier can be an email (teacher/admin, or an email-registered
    # student) or a student ID (a teacher-added student, or a student who
    # chose to register with an ID instead of an email).
    user = (
        db.query(User)
        .filter((User.email == payload.identifier) | (User.student_id == payload.identifier))
        .first()
    )
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email/ID or password.")

    token = create_access_token(subject=str(user.id), extra_claims={"role": user.role.value})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserOut)
def update_profile(
    payload: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.name is not None:
        current_user.name = payload.name
    if payload.avatar_url is not None:
        current_user.avatar_url = payload.avatar_url
    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/change-password", status_code=204)
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from the current one.")

    current_user.password_hash = hash_password(payload.new_password)
    current_user.must_change_password = False
    db.commit()
    return None


@router.post("/forgot-password/verify", status_code=204)
def forgot_password_verify(
    payload: ForgotPasswordVerifyRequest,
    db: Session = Depends(get_db),
):
    """Verify identity for in-app password reset (no email is sent)."""
    _find_user_for_password_reset(
        payload.role,
        payload.email,
        payload.student_id,
        db,
    )
    return None


@router.post("/forgot-password/reset", status_code=204)
def forgot_password_reset(
    payload: ForgotPasswordResetRequest,
    db: Session = Depends(get_db),
):
    """Reset password after role-specific verification."""
    user = _find_user_for_password_reset(
        payload.role,
        payload.email,
        payload.student_id,
        db,
    )
    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    db.commit()
    return None


@router.post("/me/photo", response_model=UserOut)
async def upload_profile_photo(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Profile photo upload for all roles. For students, verifies face quality
    and registers embeddings with the AI recognition engine. For teachers/admins,
    updates their avatar URL."""
    from app.routers.recognition import attendance_system

    os.makedirs("known_students", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/profile_{current_user.id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        user_role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
        if user_role_str == "student":
            is_valid, message = attendance_system.verify_face_quality(temp_path)
            if not is_valid:
                raise HTTPException(status_code=400, detail=message)

            success, message = attendance_system.register_student_face(
                temp_path, current_user.id, current_user.name
            )
            if not success:
                raise HTTPException(status_code=400, detail=message)

            current_user.avatar_url = f"/media/known_students/{current_user.id}.jpg"
        else:
            dest_path = f"known_students/{current_user.id}.jpg"
            shutil.copyfile(temp_path, dest_path)
            current_user.avatar_url = f"/media/known_students/{current_user.id}.jpg"

        db.commit()
        db.refresh(current_user)
        return current_user
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


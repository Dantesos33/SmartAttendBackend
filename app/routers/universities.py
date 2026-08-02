from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.university import University
from app.models.user import User, UserRole
from app.schemas.university import UniversityCreate, UniversityOut
from app.core.deps import get_current_user, require_role

router = APIRouter(prefix="/universities", tags=["universities"])


@router.get("", response_model=list[UniversityOut])
def list_universities(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Every authenticated role can read the list (needed for the create-class
    university picker, the browse-classes filter, etc.) — only admins can write."""
    return db.query(University).order_by(University.name).all()


@router.post("", response_model=UniversityOut, status_code=201)
def create_university(
    payload: UniversityCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_role(UserRole.admin)),
):
    existing = db.query(University).filter(University.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="A university with this name already exists.")
    uni = University(name=payload.name)
    db.add(uni)
    db.commit()
    db.refresh(uni)
    return uni

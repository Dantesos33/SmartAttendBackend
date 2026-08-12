"""Shared SQLAlchemy query helpers to avoid N+1 lazy loads."""

from sqlalchemy.orm import Session, joinedload, selectinload

from app.models.class_ import Class, Section


CLASS_OUT_OPTIONS = (
    joinedload(Class.university),
    joinedload(Class.sections).selectinload(Section.enrollments),
)


def load_class_for_out(db: Session, class_id: int) -> Class | None:
    return (
        db.query(Class)
        .options(*CLASS_OUT_OPTIONS)
        .filter(Class.id == class_id)
        .first()
    )


def get_section_with_class(db: Session, section_id: int) -> Section | None:
    return (
        db.query(Section)
        .options(joinedload(Section.class_))
        .filter(Section.id == section_id)
        .first()
    )

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, engine, Base
from app.models.user import User, UserRole
from app.core.deps import require_role
from migrate_db import run_migrations

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/migrate", status_code=200)
def trigger_database_migration(
    current_user: User = Depends(require_role(UserRole.admin)),
):
    """Trigger manual database schema sync and migrations."""
    try:
        run_migrations()
        return {"status": "success", "message": "Database migration executed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database migration failed: {str(e)}")

import os
import sys
from pathlib import Path

# The local migration script directory is named ``alembic``. Remove the
# backend path while importing the third-party package so it is not shadowed
# by that local namespace package when this file is run directly.
backend_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(backend_dir)
os.chdir(parent_dir)
sys.path = [entry for entry in sys.path if entry not in ("", backend_dir)]
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

# Add backend directory to sys.path
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

from app.models.user import User, UserRole
from app.database import engine
from app.core.security import hash_password
from app.database import SessionLocal

def run_migrations():
    print("Starting SmartAttend database migration...")

    alembic_config = Config(str(Path(__file__).with_name("alembic.ini")))
    inspector = inspect(engine)
    if inspector.has_table("users") and not inspector.has_table("alembic_version"):
        # Adopt databases created by the former create_all-based bootstrap.
        # They already contain the baseline schema, so stamping avoids trying
        # to recreate existing tables; all future revisions still run normally.
        command.stamp(alembic_config, "head")
        print("✓ Existing schema adopted as Alembic baseline.")
    else:
        command.upgrade(alembic_config, "head")
        print("✓ Alembic migrations applied.")

    # 2. Check and seed initial admin user if no admin exists
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.role == UserRole.admin).first()
        if not admin_user:
            admin_email = os.getenv("ADMIN_EMAIL", "admin@smartattend.com")
            admin_password = os.getenv("ADMIN_PASSWORD", "Admin123!")
            print(f"Creating default admin account ({admin_email})...")
            new_admin = User(
                name="System Administrator",
                email=admin_email,
                password_hash=hash_password(admin_password),
                role=UserRole.admin,
                must_change_password=False,
            )
            db.add(new_admin)
            db.commit()
            print("✓ Default admin account created successfully.")
        else:
            print("✓ Admin account already exists.")
            
        print("✓ Database migration finished successfully.")
    except Exception as e:
        db.rollback()
        print(f"❌ Error during database migration: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    run_migrations()

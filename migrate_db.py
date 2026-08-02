import os
import sys
from sqlalchemy import inspect, text

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.database import engine, Base
from app import models  # Ensure all SQLAlchemy models are loaded
from app.models.user import User, UserRole
from app.core.security import hash_password
from app.database import SessionLocal

def run_migrations():
    print("Starting SmartAttend database migration...")
    
    # 1. Ensure all tables defined in models exist in the database
    Base.metadata.create_all(bind=engine)
    print("✓ Created/verified all database tables.")

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

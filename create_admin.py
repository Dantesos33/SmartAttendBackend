"""
Creates an admin account directly in the database. Run this from the
main-backend directory:

    python create_admin.py

Admin accounts are intentionally not self-registerable through the API
(see app/routers/auth.py) — this script is the supported way to create one.
"""
import getpass

from app.database import SessionLocal
from app.models.user import User, UserRole
from app.core.security import hash_password


def main():
    print("=== Create SmartAttend Admin Account ===")
    name = input("Name: ").strip()
    email = input("Email: ").strip().lower()
    password = getpass.getpass("Password (min 8 chars, at least 1 letter + 1 number): ")

    if len(password) < 8 or not any(c.isdigit() for c in password) or not any(c.isalpha() for c in password):
        print("Password doesn't meet the minimum strength requirement. Aborting.")
        return

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"A user with email '{email}' already exists (role: {existing.role.value}). Aborting.")
            return

        admin = User(
            name=name,
            email=email,
            password_hash=hash_password(password),
            role=UserRole.admin,
        )
        db.add(admin)
        db.commit()
        print(f"Admin account created: {email}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

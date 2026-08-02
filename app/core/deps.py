from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.security import decode_access_token
from app.models.user import User, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user


def require_role(*allowed_roles: UserRole):
    """Usage: Depends(require_role(UserRole.admin, UserRole.teacher))"""

    def dependency(current_user: User = Depends(get_current_user)) -> User:
        user_role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
        allowed_roles_str = [
            r.value if hasattr(r, "value") else str(r) for r in allowed_roles
        ]
        if user_role_str not in allowed_roles_str:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user_role_str}' is not permitted to perform this action.",
            )
        return current_user

    return dependency

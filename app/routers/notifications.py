from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.notification import Notification
from app.schemas.notification import NotificationOut, RegisterPushTokenRequest
from app.core.deps import get_current_user

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.post("/register-device", status_code=204)
def register_device(
    payload: RegisterPushTokenRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Stores/overwrites the Expo push token for the device the caller is
    currently signed in on, so future notifications can also be delivered
    as a push. Called on login/app-start when Push Notifications is enabled
    in Settings, and whenever the OS re-issues a token."""
    current_user.push_token = payload.push_token
    db.commit()
    return None


@router.delete("/register-device", status_code=204)
def unregister_device(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Clears the stored push token — called on logout and when the user
    turns Push Notifications off in Settings, so no further pushes are sent
    to this device."""
    current_user.push_token = None
    db.commit()
    return None


@router.get("", response_model=list[NotificationOut])
def my_notifications(
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Notification).filter(Notification.user_id == current_user.id)
    if unread_only:
        query = query.filter(Notification.read_at.is_(None))
    return query.order_by(Notification.created_at.desc()).all()


@router.post("/{notification_id}/read", response_model=NotificationOut)
def mark_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    n = db.query(Notification).filter(Notification.id == notification_id).first()
    if not n or n.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Notification not found.")
    n.read_at = datetime.utcnow()
    db.commit()
    db.refresh(n)
    return n


@router.post("/read-all", status_code=204)
def mark_all_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db.query(Notification).filter(
        Notification.user_id == current_user.id, Notification.read_at.is_(None)
    ).update({"read_at": datetime.utcnow()})
    db.commit()
    return None


@router.delete("/{notification_id}", status_code=204)
def delete_notification(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    n = db.query(Notification).filter(Notification.id == notification_id).first()
    if not n or n.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Notification not found.")
    db.delete(n)
    db.commit()
    return None


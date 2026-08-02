from sqlalchemy.orm import Session

from app.models.notification import Notification, NotificationType


def notify(
    db: Session,
    user_id: int,
    type_: NotificationType,
    title: str,
    body: str | None = None,
    related_id: int | None = None,
) -> Notification:
    """Creates and adds a notification to the session. Caller is responsible for
    db.commit() — kept this way so callers can batch multiple notifications
    (e.g. one per student after an attendance session) into a single commit."""
    n = Notification(
        user_id=user_id,
        type=type_,
        title=title,
        body=body,
        related_id=related_id,
    )
    db.add(n)
    return n

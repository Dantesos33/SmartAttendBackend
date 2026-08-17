import json
import urllib.request
import urllib.error

from sqlalchemy.orm import Session

from app.models.notification import Notification, NotificationType
from app.models.user import User

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


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
    (e.g. one per student after an attendance session) into a single commit.

    Also fires a best-effort push notification to the user's registered
    device (if any). Push delivery never raises: a failed/unreachable Expo
    push service must not break the in-app notification it's piggybacking on.
    """
    n = Notification(
        user_id=user_id,
        type=type_,
        title=title,
        body=body,
        related_id=related_id,
    )
    db.add(n)
    _send_push_best_effort(db, user_id, type_, title, body, related_id)
    return n


def _send_push_best_effort(
    db: Session,
    user_id: int,
    type_: NotificationType,
    title: str,
    body: str | None,
    related_id: int | None,
) -> None:
    try:
        token = (
            db.query(User.push_token)
            .filter(User.id == user_id)
            .scalar()
        )
        if not token:
            return
        send_expo_push(
            token,
            title,
            body,
            data={
                "type": type_.value if hasattr(type_, "value") else str(type_),
                "related_id": related_id,
            },
        )
    except Exception as exc:  # noqa: BLE001 — never let push delivery break a request
        print(f"Push notification skipped for user {user_id}: {exc}")


def send_expo_push(
    token: str,
    title: str,
    body: str | None = None,
    data: dict | None = None,
    timeout: float = 5.0,
) -> None:
    """Sends a single push message through Expo's push service. Expo push
    tokens look like 'ExponentPushToken[...]'; anything else (e.g. a stale
    value from a dev build without a valid token) is skipped rather than
    sent, since Expo would just reject it anyway."""
    if not token or not token.startswith("ExponentPushToken"):
        return

    payload = {
        "to": token,
        "title": title,
        "body": body or "",
        "sound": "default",
        "data": data or {},
    }
    req = urllib.request.Request(
        EXPO_PUSH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except urllib.error.URLError as exc:
        print(f"Expo push delivery failed: {exc}")

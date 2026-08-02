from datetime import datetime
from pydantic import BaseModel

from app.models.notification import NotificationType


class NotificationOut(BaseModel):
    id: int
    type: NotificationType
    title: str
    body: str | None
    related_id: int | None
    read_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True

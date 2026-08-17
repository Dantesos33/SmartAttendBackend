"""Add Expo push token on users.

Revision ID: 0003_user_push_token
Revises: 0002_user_face_encoding
Create Date: 2026-08-18
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_user_push_token"
down_revision = "0002_user_face_encoding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("push_token", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "push_token")

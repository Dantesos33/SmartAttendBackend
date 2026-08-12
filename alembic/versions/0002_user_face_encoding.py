"""Add persistent face encodings on users.

Revision ID: 0002_user_face_encoding
Revises: 0001_initial_schema
Create Date: 2026-08-13
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_user_face_encoding"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("face_encoding_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "face_encoding_json")

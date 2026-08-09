"""Create the initial SmartAttend schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-09
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    user_role = sa.Enum("admin", "teacher", "student", name="userrole")
    enrollment_status = sa.Enum("pending", "accepted", "rejected", name="enrollmentrequeststatus")
    notification_type = sa.Enum(
        "enrollment_accepted", "enrollment_rejected", "attendance_requested",
        "attendance_result", "enrollment_request_received", "low_attendance_warning",
        "enrolled_by_teacher", "leave_request_received", "leave_approved", "leave_rejected",
        name="notificationtype",
    )
    attendance_status = sa.Enum("present", "absent", "leave", name="attendancestatus")
    leave_status = sa.Enum("pending", "approved", "rejected", name="leaverequeststatus")

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=190), nullable=True),
        sa.Column("student_id", sa.String(length=50), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", user_role, nullable=False),
        sa.Column("avatar_url", sa.String(length=500), nullable=True),
        sa.Column("must_change_password", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("student_id"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_student_id", "users", ["student_id"])

    op.create_table(
        "universities",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=190), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "teacher_universities",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("teacher_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("university_id", sa.Integer(), sa.ForeignKey("universities.id"), nullable=False),
        sa.UniqueConstraint("teacher_id", "university_id", name="uq_teacher_university"),
    )

    op.create_table(
        "classes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=190), nullable=False),
        sa.Column("code", sa.String(length=30), nullable=False),
        sa.Column("subject", sa.String(length=190), nullable=True),
        sa.Column("university_id", sa.Integer(), sa.ForeignKey("universities.id"), nullable=False),
        sa.Column("teacher_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("code"),
    )

    op.create_table(
        "sections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("classes.id"), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("schedule_days", sa.String(length=100), nullable=True),
        sa.Column("start_time", sa.Time(), nullable=True),
        sa.Column("end_time", sa.Time(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "enrollments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("student_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("section_id", sa.Integer(), sa.ForeignKey("sections.id"), nullable=False),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("classes.id"), nullable=False),
        sa.Column("enrolled_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("student_id", "class_id", name="uq_student_one_section_per_class"),
    )

    op.create_table(
        "enrollment_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("student_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("section_id", sa.Integer(), sa.ForeignKey("sections.id"), nullable=False),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("classes.id"), nullable=False),
        sa.Column("status", enrollment_status, nullable=False, server_default="pending"),
        sa.Column("requested_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "attendance_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("section_id", sa.Integer(), sa.ForeignKey("sections.id"), nullable=False),
        sa.Column("taken_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("time", sa.Time(), nullable=False),
        sa.Column("present_count", sa.Integer(), server_default="0"),
        sa.Column("absent_count", sa.Integer(), server_default="0"),
        sa.Column("leave_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "attendance_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("attendance_sessions.id"), nullable=False),
        sa.Column("student_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", attendance_status, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "leave_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("student_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("section_id", sa.Integer(), sa.ForeignKey("sections.id"), nullable=False),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("classes.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column("status", leave_status, nullable=False, server_default="pending"),
        sa.Column("requested_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "notifications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("type", notification_type, nullable=False),
        sa.Column("title", sa.String(length=190), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("related_id", sa.Integer(), nullable=True),
        sa.Column("read_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_notifications_user_id", table_name="notifications")
    op.drop_table("notifications")
    op.drop_table("leave_requests")
    op.drop_table("attendance_records")
    op.drop_table("attendance_sessions")
    op.drop_table("enrollment_requests")
    op.drop_table("enrollments")
    op.drop_table("sections")
    op.drop_table("classes")
    op.drop_table("teacher_universities")
    op.drop_table("universities")
    op.drop_index("ix_users_student_id", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

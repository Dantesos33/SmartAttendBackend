"""Reporting endpoints — real aggregated attendance data for the analytics screen."""
from datetime import date, timedelta
from collections import defaultdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func

from app.database import get_db
from app.models.user import User, UserRole
from app.models.class_ import Class, Section
from app.models.enrollment import Enrollment
from app.models.attendance import AttendanceSession, AttendanceRecord, AttendanceStatus
from app.core.deps import get_current_user, require_role

router = APIRouter(prefix="/reports", tags=["reports"])


def _teacher_class_ids(db: Session, teacher_id: int) -> list[int]:
    return [c.id for c in db.query(Class.id).filter(Class.teacher_id == teacher_id).all()]


@router.get("/summary")
def get_report_summary(
    period: str = Query("weekly", enum=["weekly", "monthly", "yearly"]),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns:
    - trend: list of {label, attendance_pct} for the chosen period
    - by_class: list of {class_name, class_code, attendance_pct}
    - stats: {avg_attendance, top_class, total_sessions, total_students}
    """
    today = date.today()

    if period == "weekly":
        days = 7
        labels = [(today - timedelta(days=i)) for i in range(days - 1, -1, -1)]
        label_fmt = lambda d: d.strftime("%a")  # Mon, Tue …
    elif period == "monthly":
        days = 30
        # group into 4 weekly buckets
        labels = None
    else:  # yearly
        days = 365
        labels = None

    start_date = today - timedelta(days=days - 1)

    # Scope sessions to the current user's classes
    if current_user.role == UserRole.teacher:
        class_ids = _teacher_class_ids(db, current_user.id)
    else:
        class_ids = [c.id for c in db.query(Class.id).all()]

    if not class_ids:
        return {"trend": [], "by_class": [], "stats": {
            "avg_attendance": None, "top_class": None,
            "total_sessions": 0, "total_students": 0,
        }}

    sessions = (
        db.query(AttendanceSession)
        .join(Section, AttendanceSession.section_id == Section.id)
        .options(joinedload(AttendanceSession.section).joinedload(Section.class_))
        .filter(
            Section.class_id.in_(class_ids),
            AttendanceSession.date >= start_date,
            AttendanceSession.date <= today,
        )
        .all()
    )

    # ---------- trend ----------
    if period == "weekly":
        day_data: dict[date, list[float]] = defaultdict(list)
        for s in sessions:
            total = s.present_count + s.absent_count
            if total > 0:
                day_data[s.date].append((s.present_count / total) * 100)
        trend = []
        for d in labels:
            vals = day_data.get(d, [])
            trend.append({
                "label": label_fmt(d),
                "attendance_pct": round(sum(vals) / len(vals), 1) if vals else None,
            })
    elif period == "monthly":
        # 4 weekly buckets
        buckets: dict[int, list[float]] = defaultdict(list)
        for s in sessions:
            days_ago = (today - s.date).days
            bucket = min(3, days_ago // 7)  # 0=most recent week
            total = s.present_count + s.absent_count
            if total > 0:
                buckets[bucket].append((s.present_count / total) * 100)
        trend = []
        for i in range(3, -1, -1):
            vals = buckets.get(i, [])
            wk_label = f"W{4 - i}"
            trend.append({
                "label": wk_label,
                "attendance_pct": round(sum(vals) / len(vals), 1) if vals else None,
            })
    else:  # yearly — 12 month buckets
        month_data: dict[tuple, list[float]] = defaultdict(list)
        for s in sessions:
            key = (s.date.year, s.date.month)
            total = s.present_count + s.absent_count
            if total > 0:
                month_data[key].append((s.present_count / total) * 100)
        months = sorted(month_data.keys())[-12:]
        trend = []
        import calendar
        for key in months:
            vals = month_data[key]
            trend.append({
                "label": calendar.month_abbr[key[1]],
                "attendance_pct": round(sum(vals) / len(vals), 1) if vals else None,
            })

    # ---------- by_class ----------
    classes = db.query(Class).filter(Class.id.in_(class_ids)).all()
    enrollment_counts = dict(
        db.query(Enrollment.class_id, func.count(Enrollment.id))
        .filter(Enrollment.class_id.in_(class_ids))
        .group_by(Enrollment.class_id)
        .all()
    )
    class_data: dict[int, dict] = {}
    for cls in classes:
        cls_id = cls.id
        if not cls:
            continue
        cls_sessions = [s for s in sessions if s.section.class_id == cls_id]
        total_p = sum(s.present_count for s in cls_sessions)
        total_t = sum(s.present_count + s.absent_count for s in cls_sessions)
        class_data[cls_id] = {
            "class_id": cls_id,
            "class_name": cls.name,
            "class_code": cls.code,
            "attendance_pct": round((total_p / total_t) * 100, 1) if total_t else None,
            "total_sessions": len(cls_sessions),
            "total_students": enrollment_counts.get(cls_id, 0),
        }

    by_class = sorted(class_data.values(), key=lambda x: (x["attendance_pct"] or 0), reverse=True)

    # ---------- stats ----------
    all_pcts = [v["attendance_pct"] for v in by_class if v["attendance_pct"] is not None]
    avg_attendance = round(sum(all_pcts) / len(all_pcts), 1) if all_pcts else None
    top_class = by_class[0]["class_code"] if by_class else None
    total_sessions = len(sessions)
    total_students = sum(v["total_students"] for v in by_class)

    return {
        "trend": trend,
        "by_class": by_class,
        "stats": {
            "avg_attendance": avg_attendance,
            "top_class": top_class,
            "total_sessions": total_sessions,
            "total_students": total_students,
        },
    }


@router.get("/export-csv")
def export_attendance_csv(
    period: str = Query("weekly", enum=["weekly", "monthly", "yearly"]),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns raw attendance records as a CSV-formatted string."""
    today = date.today()
    days_map = {"weekly": 7, "monthly": 30, "yearly": 365}
    start_date = today - timedelta(days=days_map[period] - 1)

    if current_user.role == UserRole.teacher:
        class_ids = _teacher_class_ids(db, current_user.id)
    else:
        class_ids = [c.id for c in db.query(Class.id).all()]

    records = (
        db.query(AttendanceRecord)
        .join(AttendanceSession, AttendanceRecord.session_id == AttendanceSession.id)
        .join(Section, AttendanceSession.section_id == Section.id)
        .options(
            joinedload(AttendanceRecord.student),
            joinedload(AttendanceRecord.session)
            .joinedload(AttendanceSession.section)
            .joinedload(Section.class_),
        )
        .filter(
            Section.class_id.in_(class_ids),
            AttendanceSession.date >= start_date,
        )
        .all()
    )

    lines = ["Date,Class,Section,Student Name,Student ID,Status"]
    for r in records:
        s = r.session
        lines.append(
            f"{s.date},{s.section.class_.name},{s.section.name},"
            f"{r.student.name},{r.student.student_id or r.student.email},{r.status.value}"
        )
    return {"csv": "\n".join(lines)}

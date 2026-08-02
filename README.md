# SmartAttend Main Backend

This is the main API: auth, universities, classes/sections, enrollment requests,
notifications, and attendance-session storage. It is a **separate service** from
the existing face-recognition microservice (`SmartAttendBackend/backend`) — that
one stays exactly as-is and keeps doing pure AI/recognition work on
`localhost:8000`. This service runs on `localhost:8001` and owns everything
relational: users, roles, who's enrolled where, who's allowed to see what.

## How the two backends relate

```
Frontend (Expo app)
  ├─→ Main backend (this project, port 8001)
  │     auth, classes, sections, enrollments, notifications,
  │     attendance-session storage, RBAC
  │
  └─→ Recognition service (SmartAttendBackend/backend, port 8000)
        /register, /recognize — pure face-recognition, no concept of
        users/classes/roles at all
```

The typical "take attendance" flow going forward: the frontend calls the
recognition service's `/recognize` directly (as it already does) to get raw
face-match results, the teacher reviews/overrides those results on-screen, and
only the **final confirmed list** gets POSTed to this backend's
`POST /attendance/sessions` — that's the one call that actually writes to the
database and fires notifications. The recognition service itself never touches
MySQL.

## Setup

1. **Create the database** (MySQL must already be running):
   ```sql
   CREATE DATABASE smartattend CHARACTER SET utf8mb4;
   CREATE USER 'smartattend_user'@'localhost' IDENTIFIED BY 'changeme';
   GRANT ALL PRIVILEGES ON smartattend.* TO 'smartattend_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # edit .env: set DATABASE_URL to match your MySQL credentials, and set a
   # real random SECRET_KEY (e.g. `python -c "import secrets; print(secrets.token_hex(32))"`)
   ```

3. **Install dependencies** (a virtualenv is recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

4. **Run it**:
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```
   Tables are auto-created on startup for now (`Base.metadata.create_all`).
   Once the schema stabilizes, switch to Alembic migrations (already in
   `requirements.txt`) instead of relying on this — `create_all` only creates
   missing tables, it won't alter existing ones if you change a model later.

5. **Explore the API**: with the server running, open
   `http://localhost:8001/docs` for interactive Swagger docs — every endpoint
   below is listed there with a "try it out" button.

## What's enforced, and where

- **Role checks** happen server-side via `require_role(...)` in
  `app/core/deps.py` — this is the real enforcement the frontend's tab-bar
  hiding was missing. A request from a disallowed role gets a 403 regardless
  of what the client UI shows.
- **Ownership** (a teacher only sees their own classes) is a `WHERE
  teacher_id = current_user.id` filter in `routers/classes.py`, not just a
  frontend list filter.
- **One-section-per-class** is enforced twice: once at request time
  (`routers/enrollments.py`) and again at accept time, in case two pending
  requests for different sections of the same class both got approved in the
  meantime. It's also backed by a real database `UniqueConstraint` on
  `(student_id, class_id)` in the `enrollments` table as a last-resort
  safety net.

## Key endpoints

| Method | Path | Who |
|---|---|---|
| POST | `/auth/register`, `/auth/login`, GET `/auth/me` | anyone / self |
| GET, POST | `/universities` | read: anyone logged in · write: admin |
| POST `/classes`, GET `/classes/mine` | teacher only, scoped to their own classes (class requires a unique `code`, e.g. "ENG101") |
| GET `/classes/browse` | student only — all classes, all teachers |
| GET `/classes/all` | admin only — every class, optionally filtered by university |
| POST `/classes/{id}/reassign-teacher` | admin only — moves a class to a different teacher |
| POST `/enrollments/request` | student |
| GET `/enrollments/requests/pending`, POST `/enrollments/requests/{id}/accept`\|`reject` | teacher, scoped to their own classes |
| GET `/enrollments/mine` | student |
| POST `/enrollments/admin/enroll`, DELETE `/enrollments/admin/unenroll` | admin only — bypasses the request flow, still notifies the student |
| POST `/attendance/sessions` | teacher — the manual-review save step; also fires a low-attendance warning the moment a student's running percentage crosses below 75% |
| GET `/attendance/sessions/section/{id}` | teacher (owner) or enrolled student |
| GET `/attendance/low-attendance` | admin only — every (student, class) pair currently under the threshold |
| POST `/attendance/request-taking/{section_id}` | enrolled student pinging their teacher |
| GET `/notifications`, POST `/notifications/{id}/read` | self |

## Notification types

`enrollment_accepted`, `enrollment_rejected`, `enrollment_request_received`,
`attendance_requested`, `attendance_result`, `low_attendance_warning`. The last
one only fires once, at the moment a student's running attendance percentage
for a class crosses below 75% (`LOW_ATTENDANCE_THRESHOLD` in
`routers/attendance.py`) — it deliberately does not re-fire on every
subsequent session while they stay below the threshold, to avoid notification
spam.

## Not done yet (next steps)

- **Alembic migrations** — set up once the schema is stable, instead of the
  quick-start `create_all`.
- **Password reset / email verification** — not built, out of scope until asked.
- **Avatar upload** — `users.avatar_url` column exists, but there's no
  upload endpoint yet (needs file storage — local disk vs S3-style, worth
  deciding before building).
- **Wiring the frontend to this** — `src/utils/api.ts` currently only talks
  to the recognition service; it needs a second client pointed at this API,
  plus replacing the AsyncStorage mock auth (`src/utils/db.ts`) with real
  calls to `/auth/login` and `/auth/register`, and swapping the hardcoded
  `classes.tsx` mock array for `GET /classes/mine` / `GET /classes/browse`.
  That's the natural next piece of work once you're ready.

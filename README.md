# SmartAttend — Complete Project

```
frontend/                    React Native (Expo) app — the mobile client
backend/
  main-backend/               FastAPI + MySQL + face_recognition — the one to run.
  recognition-service/        Original standalone version, kept for reference only.
                              Its code is merged into main-backend/app/routers/recognition.py.
```

## Run order

1. **backend/main-backend** (port 8000):
   ```
   cd backend/main-backend
   pip install -r requirements.txt
   cp .env.example .env              # set MySQL credentials + a real SECRET_KEY
   python create_admin.py            # admin accounts can't self-register
   uvicorn app.main:app --reload --port 8000
   ```
2. **frontend**: `npm install` then `npx expo start`. Auto-detects your LAN IP from
   the Expo dev connection — nothing to hardcode.

## What changed in this pass

### Authentication rework — ID-based login for students
- Teachers/admins still log in with **email**. Students can now log in with
  **either an email or a student ID** — whichever they registered with, or
  the ID a teacher assigned them.
- `POST /auth/login` takes a single `identifier` field (matches against
  either column) instead of a hardcoded email field.
- New `AddStudentsBulkRequest` flow: a teacher enters just **name + student
  ID** (no photo, no email) for one or many students at once. A brand new ID
  creates a real account with a default password of `{id}@123`
  (`must_change_password = true`, forced to change on first login). An
  existing ID just enrolls that student instead of duplicating the account.
  Either way the student is notified and the one-section-per-class rule
  still applies.
- **Mandatory gates for students**, enforced through a single shared
  `resolvePostLoginRoute()` helper (used identically by login, register, and
  app startup, so they can't drift out of sync):
  1. Forced password change first, if `must_change_password` is set.
  2. Mandatory profile-photo verification (`POST /auth/me/photo`) — exactly
     one clear face required, verified server-side, rejected with a specific
     reason otherwise. This is also the *only* place a face gets registered
     with the recognition engine now — there's no teacher-facing "type a
     name, upload a photo" flow anymore.
  3. Only then, the real dashboard.

### Recognition engine — now keyed by real student ID
Reworked to store/match against real `student_id` integers instead of typed
name strings, so a recognition result maps directly to a real enrolled
student — no more reconciling AI output against the roster by name. Also
fixed a real bug in the process: the standalone version used to save
attendance the instant `/recognize` ran, *before* any human review — that's
gone. Recognition is now purely advisory; only `POST /attendance/sessions`
(called after the teacher reviews and edits results) persists anything.

### New: full manual attendance review
Camera → recognize → review screen now shows **every enrolled student** in
the section (not just recognized faces), pre-marked Present if the AI caught
them, "Leave" if they have an approved leave request for that date, Absent
otherwise — and every row is tappable to cycle between the three states
before saving.

### New: Leave Requests
Students can request leave for a class session; teachers accept/reject
(`/leave-requests/*`). A new `leave` attendance status is correctly excluded
from attendance-percentage math everywhere it's calculated (this took two
passes to get right — the first fix only covered the low-attendance-warning
path, a second bug remained in the student-history endpoint's own separate
calculation until a test caught it).

### New: Notifications tab, Browse Classes, class edit/delete
- **Notifications** is now a real tab for every role.
- **Browse Classes** — the student self-service enrollment screen (search by
  class/teacher, filter by university, request a section) — was designed
  earlier but never actually built as a screen; it exists now.
- Classes can be edited and deleted (`PATCH`/`DELETE /classes/{id}`) — deletion
  also cleans up enrollment/leave requests referencing the class first, since
  those don't cascade automatically and would otherwise violate a foreign-key
  constraint on real MySQL (SQLite silently allowed it, which would have
  hidden the bug).

### Bug fixes from your numbered list
1. **Home tab not active after login** — `login.tsx`/`register.tsx` were
   routing straight to hidden non-tab routes (`/admindashboard`,
   `/studentdashboard`) instead of the real `dashboard` tab. Fixed.
2. **Demo admin not logging in** — not a bug; admin accounts can't
   self-register by design. Run `create_admin.py`.
3. **Wrong error messages** — new `ErrorModal` component (theme-aware, works
   in dark mode unlike the native `Alert`), wired into login/register.
5. **No manual attendance editing** — the review screen rework above.
6 & 11. **Dashboard/history not updating after attendance** — the save flow
   now `router.replace`s back to the dashboard tab, which re-runs its
   `useFocusEffect` fetch and shows the just-saved session.
7. **Teacher seeing other teachers' students** — `/dashboard/stats` and
   `/classes/{id}/roster` were already ownership-scoped; the attendance
   review screen now also only ever shows the real roster for the section
   being captured.
8. **Camera FAB skipping class selection** — now goes to the classes list
   first.
9. **Quick Actions all landing on the same screen** — now carry an `intent`
   so picking a class jumps straight to the right destination.
10. **No way to edit/delete a class** — added.

## Known remaining gap
`reports.tsx`'s charts are still on sample data — the one screen not yet
wired to real attendance history.

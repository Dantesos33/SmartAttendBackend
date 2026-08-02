from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.database import Base, engine
from app import models  # noqa: F401 — ensures all models are registered on Base before create_all
from app.config import settings
from app.routers import auth, universities, classes, enrollments, notifications, attendance, dashboard, recognition, leave, admin, reports

app = FastAPI(title="SmartAttend Main API")


origins = ["*"] if settings.CORS_ORIGINS == "*" else [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Quick-start table creation. For real schema changes going forward, switch
    # to Alembic migrations (already in requirements.txt) instead of relying on
    # create_all, which won't alter existing tables.
    Base.metadata.create_all(bind=engine)


@app.get("/")
def read_root():
    return {"message": "SmartAttend main API is running."}


os.makedirs("known_students", exist_ok=True)
app.mount("/media/known_students", StaticFiles(directory="known_students"), name="known_students_media")


app.include_router(auth.router)
app.include_router(universities.router)
app.include_router(classes.router)
app.include_router(enrollments.router)
app.include_router(notifications.router)
app.include_router(attendance.router)
app.include_router(dashboard.router)
app.include_router(recognition.router)
app.include_router(leave.router)
app.include_router(admin.router)
app.include_router(reports.router)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

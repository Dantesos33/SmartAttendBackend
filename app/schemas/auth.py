from pydantic import BaseModel, EmailStr, field_validator, model_validator

from app.models.user import UserRole


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr | None = None
    student_id: str | None = None
    password: str
    role: UserRole = UserRole.student

    @model_validator(mode="after")
    def require_an_identifier(self):
        if self.role in (UserRole.teacher, UserRole.admin) and not self.email:
            raise ValueError("Email is required for teacher and admin accounts.")
        if self.role == UserRole.student and not self.email and not self.student_id:
            raise ValueError("Provide either an email or a student ID to register.")
        return self

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number.")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter.")
        return v

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters long.")
        return v


class LoginRequest(BaseModel):
    # Deliberately a plain string, not EmailStr — this can be either an email
    # (teacher/admin, or a student who registered with one) or a student ID
    # (a student added by a teacher, or who chose to self-register with an ID
    # instead of an email). The login endpoint tries both columns.
    identifier: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number.")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter.")
        return v


class UpdateProfileRequest(BaseModel):
    name: str | None = None
    avatar_url: str | None = None

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if len(v) < 2:
                raise ValueError("Name must be at least 2 characters long.")
        return v


class UserOut(BaseModel):
    id: int
    name: str
    email: str | None = None
    student_id: str | None = None
    role: UserRole
    avatar_url: str | None = None
    must_change_password: bool = False

    class Config:
        from_attributes = True

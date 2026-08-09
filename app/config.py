from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+pymysql://smartattend_user:changeme@localhost:3306/smartattend"
    SECRET_KEY: str = "change-this-to-a-long-random-string"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ALGORITHM: str = "HS256"
    RECOGNITION_SERVICE_URL: str = "http://localhost:8000"
    CORS_ORIGINS: str = "*"
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_production_security(self):
        if self.ENVIRONMENT.lower() in {"production", "prod"}:
            if len(self.SECRET_KEY) < 32 or self.SECRET_KEY == "change-this-to-a-long-random-string":
                raise ValueError("SECRET_KEY must be a random value of at least 32 characters in production.")
            if self.CORS_ORIGINS.strip() == "*":
                raise ValueError("CORS_ORIGINS must list explicit origins in production.")
        return self


settings = Settings()

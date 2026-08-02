from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+pymysql://smartattend_user:changeme@localhost:3306/smartattend"
    SECRET_KEY: str = "change-this-to-a-long-random-string"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ALGORITHM: str = "HS256"
    RECOGNITION_SERVICE_URL: str = "http://localhost:8000"
    CORS_ORIGINS: str = "*"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

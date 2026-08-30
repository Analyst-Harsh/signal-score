"""Shared runtime settings, loaded once from the environment / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

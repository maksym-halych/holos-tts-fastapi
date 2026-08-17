from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HOLOS_")

    weights_path: Path = Path("/app/weights/holos.pt")
    voices_path: Path = Path("/app/weights/voices.pt")
    device: str = "cpu"
    default_voice: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    def validate_startup(self) -> None:
        if not self.weights_path.is_file():
            raise ValueError(
                f"HOLOS_WEIGHTS_PATH does not point to an existing file: {self.weights_path}"
            )
        if not self.voices_path.is_file():
            raise ValueError(
                f"HOLOS_VOICES_PATH does not point to an existing file: {self.voices_path}"
            )
        try:
            torch.device(self.device)
        except RuntimeError as exc:
            raise ValueError(f"HOLOS_DEVICE is invalid: {self.device!r} ({exc})") from exc


settings = Settings()

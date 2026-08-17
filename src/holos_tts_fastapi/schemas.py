from typing import Literal

from pydantic import BaseModel, Field, field_validator

from holos_tts_fastapi.audio import SUPPORTED_FORMATS

ResponseFormat = Literal["mp3", "wav", "opus", "flac", "aac", "pcm"]


class SpeechRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    input: str
    model: str | None = None
    voice: str | None = None
    response_format: ResponseFormat = "mp3"
    speed: float = 1.0
    stream: bool = True

    @field_validator("input")
    @classmethod
    def input_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input must not be empty")
        return value

    @field_validator("speed")
    @classmethod
    def speed_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("speed must be a positive number")
        return value

    @field_validator("response_format")
    @classmethod
    def response_format_must_be_supported(cls, value: str) -> str:
        if value not in SUPPORTED_FORMATS:
            raise ValueError(f"response_format must be one of {', '.join(SUPPORTED_FORMATS)}")
        return value


class VoicesResponse(BaseModel):
    voices: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    model_loaded: bool = Field(description="Whether the HolosTTS model has finished loading")

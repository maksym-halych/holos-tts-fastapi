from typing import Literal

from pydantic import BaseModel, Field, field_validator

from holos_tts_fastapi.audio import SUPPORTED_FORMATS

ResponseFormat = Literal["mp3", "wav", "opus", "flac", "aac", "pcm"]

# Same range as the OpenAI speech API. The model divides predicted durations by
# speed and rounds them with no lower bound, so much higher values collapse
# whole sentences to zero frames, which it rejects.
MIN_SPEED = 0.25
MAX_SPEED = 4.0


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
    def speed_must_be_in_range(cls, value: float) -> float:
        if not MIN_SPEED <= value <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
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

from collections.abc import AsyncIterator

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from holos_tts_fastapi.api import get_engine, router, validation_error_handler
from holos_tts_fastapi.engine import UnknownVoiceError

FAKE_VOICES = ["voice_a", "voice_b"]


class FakeEngine:
    def __init__(self, ready: bool = True):
        self.ready = ready
        self.calls: list[tuple[str, str | None, float]] = []

    def list_voices(self) -> list[str]:
        return FAKE_VOICES

    def resolve_voice(self, voice: str | None) -> str:
        name = voice or FAKE_VOICES[0]
        if name not in FAKE_VOICES:
            raise UnknownVoiceError(name, FAKE_VOICES)
        return name

    async def generate_chunks(self, text: str, voice: str | None, speed: float) -> AsyncIterator[np.ndarray]:
        self.calls.append((text, voice, speed))
        # Two short deterministic chunks, enough to exercise multi-chunk streaming.
        for _ in range(2):
            yield np.zeros(2400, dtype=np.float32)


def make_client(engine: FakeEngine) -> TestClient:
    app = FastAPI()
    app.state.engine = engine
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(router)
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


def test_health_reports_ready():
    client = make_client(FakeEngine(ready=True))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_loading():
    client = make_client(FakeEngine(ready=False))
    resp = client.get("/health")
    assert resp.json()["status"] == "loading"


def test_list_voices():
    client = make_client(FakeEngine())
    resp = client.get("/v1/audio/voices")
    assert resp.status_code == 200
    assert resp.json() == {"voices": FAKE_VOICES}


def test_minimal_speech_request_succeeds():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "Привіт"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0


def test_model_field_is_accepted_and_ignored():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "Привіт", "model": "tts-1"})
    assert resp.status_code == 200


def test_response_format_sets_content_type():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "Привіт", "response_format": "wav", "stream": False})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content.startswith(b"RIFF")


def test_empty_input_is_rejected():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "   "})
    assert resp.status_code == 422


def test_missing_input_is_rejected():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={})
    assert resp.status_code == 422


def test_unsupported_format_is_rejected():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "Привіт", "response_format": "wma"})
    assert resp.status_code == 422


@pytest.mark.parametrize("speed", [0, -1, 0.24, 4.01, 100])
def test_out_of_range_speed_is_rejected(speed):
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "Привіт", "speed": speed})
    assert resp.status_code == 422
    assert "speed" in resp.text


@pytest.mark.parametrize("raw_speed", ["1e999", "NaN", "Infinity"])
def test_non_finite_speed_is_rejected(raw_speed):
    client = make_client(FakeEngine())
    body = '{"input": "Привіт", "speed": ' + raw_speed + "}"
    resp = client.post("/v1/audio/speech", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "speed"]


@pytest.mark.parametrize("speed", [0.25, 1.0, 4.0])
def test_speed_range_limits_are_accepted(speed):
    engine = FakeEngine()
    client = make_client(engine)
    resp = client.post("/v1/audio/speech", json={"input": "Привіт", "speed": speed, "response_format": "pcm"})
    assert resp.status_code == 200
    assert engine.calls[-1][2] == speed


def test_unknown_voice_is_rejected_with_valid_names_listed():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "Привіт", "voice": "alloy"})
    assert resp.status_code == 422
    assert "voice_a" in resp.json()["detail"]


def test_long_input_is_accepted():
    client = make_client(FakeEngine())
    long_text = "Привіт. " * 5000
    resp = client.post("/v1/audio/speech", json={"input": long_text, "stream": False})
    assert resp.status_code == 200


def test_buffered_response_is_complete():
    engine = FakeEngine()
    client = make_client(engine)
    resp = client.post("/v1/audio/speech", json={"input": "Привіт", "response_format": "pcm", "stream": False})
    assert resp.status_code == 200
    assert len(resp.content) == 2 * 2400 * 2  # 2 chunks * samples * 2 bytes/sample


def test_streaming_is_default():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/speech", json={"input": "Привіт"})
    assert resp.status_code == 200


def test_voice_combine_route_absent():
    client = make_client(FakeEngine())
    resp = client.post("/v1/audio/voices/combine", json={})
    assert resp.status_code == 404


def test_captioned_speech_route_absent():
    client = make_client(FakeEngine())
    resp = client.post("/dev/captioned_speech", json={})
    assert resp.status_code == 404


def test_phonemize_route_absent():
    client = make_client(FakeEngine())
    resp = client.post("/dev/phonemize", json={})
    assert resp.status_code == 404


def test_dialogue_route_absent():
    client = make_client(FakeEngine())
    resp = client.post("/dev/dialogue", json={})
    assert resp.status_code == 404

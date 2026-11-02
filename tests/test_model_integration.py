"""Runs the real engine against a real checkpoint, unlike test_api.py's FakeEngine.

This is what catches a checkpoint that no longer matches the holos-tts
architecture. Skipped unless HOLOS_WEIGHTS_PATH and HOLOS_VOICES_PATH point at
real files; the image smoke test in CI covers the same ground on every build.
"""
import asyncio

import numpy as np
import pytest

from holos_tts_fastapi.audio import FIXED_SCALE
from holos_tts_fastapi.config import Settings
from holos_tts_fastapi.engine import SAMPLE_RATE, SynthesisEngine

SENTENCE = "Будь ласка, зателефонуйте мені завтра після обіду, коли матимете вільну хвилинку."

settings = Settings()
pytestmark = pytest.mark.skipif(
    not (settings.weights_path.is_file() and settings.voices_path.is_file()),
    reason="HOLOS_WEIGHTS_PATH and HOLOS_VOICES_PATH must point at real files",
)


@pytest.fixture(scope="module")
def engine() -> SynthesisEngine:
    engine = SynthesisEngine(settings)
    engine.load()
    return engine


def synthesize(engine: SynthesisEngine, text: str, voice: str | None = None, speed: float = 1.0) -> np.ndarray:
    async def collect() -> list[np.ndarray]:
        return [chunk async for chunk in engine.generate_chunks(text, voice, speed)]

    chunks = asyncio.run(collect())
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def test_checkpoint_loads_and_exposes_voices(engine):
    assert engine.ready
    assert "Speaker_0" in engine.list_voices()


def test_sentence_produces_plausible_audio(engine):
    wav = synthesize(engine, SENTENCE, "Speaker_0")
    assert wav.dtype == np.float32
    assert np.isfinite(wav).all()
    assert 2 < wav.size / SAMPLE_RATE < 10


def test_every_voice_stays_below_the_clipping_point(engine):
    for voice in engine.list_voices():
        wav = synthesize(engine, SENTENCE, voice)
        assert np.abs(wav).max() * FIXED_SCALE <= 32767, voice


def test_speed_changes_duration(engine):
    slow = synthesize(engine, SENTENCE, "Speaker_0", speed=0.25)
    normal = synthesize(engine, SENTENCE, "Speaker_0", speed=1.0)
    fast = synthesize(engine, SENTENCE, "Speaker_0", speed=4.0)
    assert slow.size > normal.size > fast.size > 0


@pytest.mark.parametrize("text", ["в", "ш", ".", "7"])
def test_shortest_chunks_survive_top_speed_on_every_voice(engine, text):
    for voice in engine.list_voices():
        assert synthesize(engine, text, voice, speed=4.0).size > 0, voice

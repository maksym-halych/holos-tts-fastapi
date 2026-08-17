import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from holos_tts_fastapi.config import Settings
from holos_tts_fastapi.frontend import text_to_ipa_chunks
from holos_tts_fastapi.nn.models import HolosTTS

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class UnknownVoiceError(ValueError):
    def __init__(self, voice: str, known_voices: list[str]):
        self.voice = voice
        self.known_voices = known_voices
        super().__init__(f"Unknown voice {voice!r}. Available voices: {', '.join(known_voices)}")


class SynthesisEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model: HolosTTS | None = None
        self._lock = asyncio.Lock()

    def load(self) -> None:
        logger.info("Loading HolosTTS model from %s (device=%s)", self._settings.weights_path, self._settings.device)
        self._model = HolosTTS(
            weights_path=str(self._settings.weights_path),
            device=self._settings.device,
            voices_path=str(self._settings.voices_path),
        )
        logger.info("Model loaded. Voices: %s", ", ".join(self._model.list_voices()))

    @property
    def ready(self) -> bool:
        return self._model is not None

    def list_voices(self) -> list[str]:
        assert self._model is not None
        return self._model.list_voices()

    def resolve_voice(self, voice: str | None) -> str:
        assert self._model is not None
        known_voices = self._model.list_voices()
        name = voice or self._settings.default_voice or known_voices[0]
        if name not in self._model.voices:
            raise UnknownVoiceError(name, known_voices)
        return name

    async def generate_chunks(self, text: str, voice: str | None, speed: float) -> AsyncIterator[np.ndarray]:
        resolved_voice = self.resolve_voice(voice)
        for ipa_chunk in text_to_ipa_chunks(text):
            async with self._lock:
                waveform = await asyncio.to_thread(self._generate_one, ipa_chunk, resolved_voice, speed)
            yield waveform

    def _generate_one(self, ipa_chunk: str, voice: str, speed: float) -> np.ndarray:
        assert self._model is not None
        wav = self._model.generate(ipa_chunk, voice, speed)
        return wav.cpu().numpy().reshape(-1)

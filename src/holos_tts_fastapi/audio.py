import asyncio
import logging
import struct
from collections.abc import AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8

# Fixed scale applied to the model's float waveform before int16 conversion.
# Chosen (rather than per-chunk peak normalization) so loudness stays
# consistent across streamed chunks; see design.md for rationale. Chunk peaks
# across varied Ukrainian sentences and all 27 presets measure 0.28-0.59 with
# the pinned weights (median 0.43), so 50000 leaves headroom to ~0.65 before
# clipping while staying comfortably audible. Re-measure when the weights change.
FIXED_SCALE = 50000.0

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

SUPPORTED_FORMATS = tuple(MEDIA_TYPES)
NATIVE_FORMATS = {"pcm", "wav"}

_FFMPEG_OUTPUT_ARGS = {
    "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-q:a", "2"],
    "opus": ["-f", "opus", "-c:a", "libopus"],
    "flac": ["-f", "flac", "-c:a", "flac"],
    "aac": ["-f", "adts", "-c:a", "aac"],
}


def to_pcm16(waveform: np.ndarray, scale: float = FIXED_SCALE) -> np.ndarray:
    """Convert a float waveform to 16-bit PCM using a fixed scale factor, clipping instead of wrapping."""
    scaled = waveform.astype(np.float32) * scale
    clipped = np.clip(scaled, -32768, 32767)
    return clipped.astype(np.int16)


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return samples.astype("<i2").tobytes()


def wav_header(num_samples: int | None) -> bytes:
    """Build a RIFF/WAVE header for 16-bit mono PCM at SAMPLE_RATE.

    When num_samples is None (streaming, total length unknown), the RIFF
    and data chunk sizes are set to the placeholder 0xFFFFFFFF, which
    standard players and decoders accept and read until EOF.
    """
    byte_rate = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
    block_align = CHANNELS * BYTES_PER_SAMPLE

    if num_samples is None:
        data_size = 0xFFFFFFFF
        riff_size = 0xFFFFFFFF
    else:
        data_size = num_samples * BYTES_PER_SAMPLE
        riff_size = 36 + data_size

    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack("<H", 1),  # PCM
            struct.pack("<H", CHANNELS),
            struct.pack("<I", SAMPLE_RATE),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", BITS_PER_SAMPLE),
            b"data",
            struct.pack("<I", data_size),
        ]
    )


class FFmpegEncoder:
    """Pipes raw PCM into an ffmpeg subprocess and yields encoded bytes incrementally."""

    def __init__(self, response_format: str):
        if response_format not in _FFMPEG_OUTPUT_ARGS:
            raise ValueError(f"No ffmpeg encoder for format {response_format!r}")
        self._args = _FFMPEG_OUTPUT_ARGS[response_format]
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_chunks: list[bytes] = []

    async def __aenter__(self) -> "FFmpegEncoder":
        self._process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-loglevel", "error",
            "-f", "s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-i", "pipe:0",
            *self._args,
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.ensure_future(self._collect_stderr())
        return self

    async def _collect_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while True:
            chunk = await self._process.stderr.read(4096)
            if not chunk:
                break
            self._stderr_chunks.append(chunk)

    async def write(self, data: bytes) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def close_input(self) -> None:
        assert self._process is not None and self._process.stdin is not None
        if not self._process.stdin.is_closing():
            self._process.stdin.close()

    async def read_output(self) -> AsyncIterator[bytes]:
        assert self._process is not None and self._process.stdout is not None
        while True:
            chunk = await self._process.stdout.read(4096)
            if not chunk:
                break
            yield chunk

    async def wait(self) -> int:
        assert self._process is not None
        returncode = await self._process.wait()
        if self._stderr_task is not None:
            await self._stderr_task
        return returncode

    def stderr_text(self) -> str:
        return b"".join(self._stderr_chunks).decode(errors="replace")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._process is None:
            return
        if self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()


async def encode_stream(
    pcm_chunks: AsyncIterator[bytes], response_format: str
) -> AsyncIterator[bytes]:
    """Encode a stream of raw PCM byte chunks into the requested container format, streaming output."""
    if response_format == "pcm":
        async for chunk in pcm_chunks:
            yield chunk
        return

    if response_format == "wav":
        yield wav_header(None)
        async for chunk in pcm_chunks:
            yield chunk
        return

    async with FFmpegEncoder(response_format) as encoder:
        async def feed() -> None:
            try:
                async for chunk in pcm_chunks:
                    await encoder.write(chunk)
            finally:
                await encoder.close_input()

        feed_task = asyncio.ensure_future(feed())
        try:
            async for out_chunk in encoder.read_output():
                yield out_chunk
        finally:
            if not feed_task.done():
                feed_task.cancel()
            try:
                await feed_task
            except (asyncio.CancelledError, Exception):
                pass

        returncode = await encoder.wait()
        if returncode != 0:
            stderr = encoder.stderr_text()
            logger.error("ffmpeg exited with code %s while encoding %s: %s", returncode, response_format, stderr)
            raise RuntimeError(f"ffmpeg encoding to {response_format} failed (exit {returncode}): {stderr}")


async def encode_buffered(pcm_chunks: AsyncIterator[bytes], response_format: str) -> bytes:
    """Encode a stream of raw PCM byte chunks into a single complete buffer, with accurate length where the format allows."""
    if response_format == "wav":
        pcm_bytes = b"".join([chunk async for chunk in pcm_chunks])
        num_samples = len(pcm_bytes) // BYTES_PER_SAMPLE
        return wav_header(num_samples) + pcm_bytes

    return b"".join([chunk async for chunk in encode_stream(pcm_chunks, response_format)])

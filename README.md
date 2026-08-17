# Holos TTS FastAPI

An OpenAI-compatible speech synthesis HTTP API for [HolosTTS](https://github.com/patriotyk/HolosTTS), a Ukrainian text-to-speech model. Point an unmodified OpenAI SDK client at this server and get Ukrainian speech back.

This project wraps the [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) service shape (`POST /v1/audio/speech`, `GET /v1/audio/voices`, streaming, six output formats) around the HolosTTS checkpoint, and lifts the Ukrainian text→IPA frontend out of HolosTTS's Gradio demo into a first-class module.

## Quickstart

### Docker (recommended)

```bash
docker build -t holos-tts-fastapi .
docker run -p 8000:8000 holos-tts-fastapi
```

The image bakes in the model weights at build time, so the container needs no network access at runtime.

### Local (uv)

```bash
uv sync
export HOLOS_WEIGHTS_PATH=/path/to/holos.pt
export HOLOS_VOICES_PATH=/path/to/voices.pt
uv run uvicorn holos_tts_fastapi.api:app --host 0.0.0.0 --port 8000
```

Weights can be downloaded from [huggingface.co/patriotyk/HolosTTS](https://huggingface.co/patriotyk/HolosTTS) (`holos.pt`, `voices.pt`; skip `voice_encoder.pt`, it's not used).

### Try it with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with client.audio.speech.with_streaming_response.create(
    model="tts-1",  # ignored — exactly one model is served
    voice="Speaker_0",
    input="Привіт світе!",
) as response:
    response.stream_to_file("output.mp3")
```

## API reference

### `POST /v1/audio/speech`

Synthesizes speech from text. Request body:

| Field             | Type    | Default | Notes                                                            |
|-------------------|---------|---------|-------------------------------------------------------------------|
| `input`           | string  | —       | Required, non-empty. Ukrainian text.                              |
| `model`           | string  | —       | Accepted and ignored (exactly one model is served).               |
| `voice`           | string  | server default | A HolosTTS preset name from `GET /v1/audio/voices`.        |
| `response_format` | string  | `mp3`   | One of `mp3`, `wav`, `opus`, `flac`, `aac`, `pcm`.                 |
| `speed`           | number  | `1.0`   | Must be positive. `>1.0` is faster, `<1.0` is slower.              |
| `stream`          | boolean | `true`  | If `false`, returns the complete audio as one buffered response.  |

Response body is raw audio bytes with `Content-Type` matching `response_format` (`audio/mpeg`, `audio/wav`, `audio/opus`, `audio/flac`, `audio/aac`, `audio/pcm`).

There is no maximum input length, request timeout, or concurrency cap — this server targets a single trusted client on a private network.

### `GET /v1/audio/voices`

Returns `{"voices": ["Speaker_0", "Speaker_1", ...]}` — the preset names in `voices.pt`, usable directly as the `voice` field.

### `GET /health`

Returns `{"status": "ok" | "loading", "model_loaded": bool}`.

## Configuration

All configuration is via environment variables (prefix `HOLOS_`):

| Variable              | Default                    | Description                                  |
|-----------------------|-----------------------------|-----------------------------------------------|
| `HOLOS_WEIGHTS_PATH`  | `/app/weights/holos.pt`    | Path to the HolosTTS checkpoint.              |
| `HOLOS_VOICES_PATH`   | `/app/weights/voices.pt`   | Path to the voice presets file.               |
| `HOLOS_DEVICE`        | `cpu`                      | PyTorch device (image ships CPU-only wheels). |
| `HOLOS_DEFAULT_VOICE` | first voice in `voices.pt` | Voice used when a request omits `voice`.      |
| `HOLOS_HOST`          | `0.0.0.0`                  | Bind host.                                    |
| `HOLOS_PORT`          | `8000`                     | Bind port.                                    |
| `HOLOS_LOG_LEVEL`     | `info`                     | Uvicorn log level.                            |

Startup fails fast with a clear error if the configured weights/voices path doesn't exist or the device string is invalid.

## v1 limitations

- **No text normalization.** Numbers, dates, currency, and abbreviations (`2026`, `грн`) are passed to the phonemizer as-is and will be mispronounced or garbled.
- **Ukrainian only.** No other language is supported.
- **No voice cloning or mixing.** Only the presets baked into `voices.pt` are available; HolosTTS's `VoiceEncoder` and voice-combination are not exposed.
- **CPU only.** The shipped image installs CPU-only PyTorch wheels; GPU inference is not wired up.
- **No auth, rate limiting, or quotas.** Designed for a single trusted client on a private network.
- **FLAC streaming has a metadata caveat.** When streamed, FLAC's header cannot declare total sample count or an MD5 checksum (unknown until the stream ends). Most decoders handle this fine; buffered (`stream: false`) FLAC does not have this limitation issue since the header is still written with unknown-length placeholders regardless — use `wav` or `pcm` if exact-length FLAC metadata matters.
- **Streaming granularity is per sentence-group,** not per word — the model is non-autoregressive and produces a complete waveform per chunk, so time-to-first-byte is bounded below by one chunk's synthesis time.

## Development

```bash
uv sync
uv run pytest
```

Text frontend tests (`tests/test_frontend.py`) and API contract tests (`tests/test_api.py`) run without model weights. End-to-end synthesis requires `HOLOS_WEIGHTS_PATH`/`HOLOS_VOICES_PATH` pointing at real files.

# Holos TTS FastAPI

An OpenAI-compatible speech synthesis HTTP API for [HolosTTS](https://github.com/patriotyk/HolosTTS), a Ukrainian text-to-speech model. Point an unmodified OpenAI SDK client at this server and get Ukrainian speech back.

This project wraps the [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) service shape (`POST /v1/audio/speech`, `GET /v1/audio/voices`, streaming, six output formats) around the HolosTTS checkpoint, and lifts the Ukrainian text→IPA frontend out of HolosTTS's Gradio demo into a first-class module. The model itself comes from the [`holos-tts`](https://pypi.org/project/holos-tts/) package on PyPI.

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
uv run hf download patriotyk/HolosTTS holos.pt voices.pt --revision f4ab024ff5798b07c921dbf16e044c184b19103e --local-dir weights
export HOLOS_WEIGHTS_PATH=$PWD/weights/holos.pt
export HOLOS_VOICES_PATH=$PWD/weights/voices.pt
uv run uvicorn holos_tts_fastapi.api:app --host 0.0.0.0 --port 8000
```

The `--revision` matters: the checkpoint has to match the model architecture in the installed `holos-tts` version, and newer weights on [huggingface.co/patriotyk/HolosTTS](https://huggingface.co/patriotyk/HolosTTS) may fail to load. `voice_encoder.pt` isn't used, so it isn't downloaded.

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
| `speed`           | number  | `1.0`   | From `0.25` to `4.0`. `>1.0` is faster, `<1.0` is slower.          |
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
- **Very short chunks may ignore high `speed`.** Near `4.0`, a chunk of a single consonant or only punctuation can round to no audio on some voices, so that chunk is synthesized at `1.0` instead.
- **Streaming granularity is per sentence-group,** not per word — the server synthesizes a complete waveform per chunk, so time-to-first-byte is bounded below by one chunk's synthesis time.

## Licensing

HolosTTS code and weights are MIT-licensed. Its author notes that the preset voices were used without their owners' consent, so commercial use requires agreement with the voice owners. Upstream's other options, cloned or mixed voices, aren't available through this server. See [upstream's license notes](https://github.com/patriotyk/HolosTTS#license).

## Development

```bash
uv sync
uv run pytest
```

Text frontend tests (`tests/test_frontend.py`) and API contract tests (`tests/test_api.py`) run without model weights. Real-model tests (`tests/test_model_integration.py`) are skipped unless `HOLOS_WEIGHTS_PATH`/`HOLOS_VOICES_PATH` point at real files.

CI runs the tests, then builds the image and runs `scripts/smoke_test_image.sh` before pushing. The script boots the image with no network access and synthesizes speech in every format, which is what proves the baked-in checkpoint loads.

### Updating the model

The `holos-tts` version and the Hugging Face weights revision have to move together. To update:

1. Bump `holos-tts` in `pyproject.toml` and run `uv lock`.
2. Update `HOLOS_WEIGHTS_REVISION` in the `Dockerfile` and the `--revision` in the Quickstart above.
3. Download those weights and run `uv run pytest` with `HOLOS_WEIGHTS_PATH`/`HOLOS_VOICES_PATH` set.
4. Re-measure chunk peaks against `FIXED_SCALE` in `src/holos_tts_fastapi/audio.py`, since new weights can change output levels.

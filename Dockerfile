FROM python:3.10-slim-bookworm

# 1. System packages (least frequently changing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# 2. Dependency layer: install from the lockfile only, without the project
# itself, so this layer is cached across application source edits.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 3. Model weights layer: baked in at build time so the container needs no
# network access at runtime. Placed before the source copy so editing
# application code does not re-trigger a 541MB download. voice_encoder.pt
# is deliberately not downloaded (voice cloning is out of scope).
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='patriotyk/HolosTTS', filename='holos.pt', local_dir='/app/weights'); \
hf_hub_download(repo_id='patriotyk/HolosTTS', filename='voices.pt', local_dir='/app/weights')"

# The Ukrainian word-stress dictionary (ukrainian-word-stress) constructs a
# stanza NLP pipeline that otherwise downloads its Ukrainian model files from
# GitHub on first use. Pre-download them at build time so the Stressifier
# singleton (frontend.py) never needs network access at runtime.
RUN python -c "import stanza; stanza.download('uk', processors='tokenize,pos,mwt')"

# 4. Application source last (most frequently changing)
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV HOLOS_WEIGHTS_PATH=/app/weights/holos.pt \
    HOLOS_VOICES_PATH=/app/weights/voices.pt \
    HOLOS_HOST=0.0.0.0 \
    HOLOS_PORT=8000 \
    HOLOS_DEVICE=cpu \
    HOLOS_LOG_LEVEL=info

EXPOSE 8000

CMD exec uvicorn holos_tts_fastapi.api:app --host ${HOLOS_HOST} --port ${HOLOS_PORT} --log-level ${HOLOS_LOG_LEVEL}

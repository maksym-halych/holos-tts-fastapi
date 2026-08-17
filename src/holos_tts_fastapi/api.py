import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from holos_tts_fastapi.audio import MEDIA_TYPES, encode_buffered, encode_stream, pcm16_bytes, to_pcm16
from holos_tts_fastapi.config import settings
from holos_tts_fastapi.engine import SynthesisEngine, UnknownVoiceError
from holos_tts_fastapi.schemas import HealthResponse, SpeechRequest, VoicesResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def get_engine(request: Request) -> SynthesisEngine:
    return request.app.state.engine


@router.get("/health", response_model=HealthResponse)
async def health(engine: SynthesisEngine = Depends(get_engine)) -> HealthResponse:
    return HealthResponse(status="ok" if engine.ready else "loading", model_loaded=engine.ready)


@router.get("/v1/audio/voices", response_model=VoicesResponse)
async def list_voices(engine: SynthesisEngine = Depends(get_engine)) -> VoicesResponse:
    return VoicesResponse(voices=engine.list_voices())


@router.post("/v1/audio/speech")
async def create_speech(
    request: SpeechRequest,
    http_request: Request,
    engine: SynthesisEngine = Depends(get_engine),
):
    try:
        resolved_voice = engine.resolve_voice(request.voice)
    except UnknownVoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    media_type = MEDIA_TYPES[request.response_format]

    async def pcm_chunks():
        async for waveform in engine.generate_chunks(request.input, resolved_voice, request.speed):
            if await http_request.is_disconnected():
                logger.info("client disconnected, stopping synthesis")
                break
            yield pcm16_bytes(to_pcm16(waveform))

    if not request.stream:
        try:
            data = await encode_buffered(pcm_chunks(), request.response_format)
        except Exception as exc:
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail="synthesis failed") from exc
        return Response(content=data, media_type=media_type)

    encoded = encode_stream(pcm_chunks(), request.response_format)
    try:
        first_chunk = await encoded.__anext__()
    except StopAsyncIteration:
        first_chunk = None
    except Exception as exc:
        logger.exception("synthesis failed before streaming began")
        raise HTTPException(status_code=500, detail="synthesis failed") from exc

    async def byte_stream():
        if first_chunk is not None:
            yield first_chunk
        try:
            async for chunk in encoded:
                yield chunk
        except Exception:
            logger.exception("synthesis failed mid-stream, terminating stream")
            return

    return StreamingResponse(byte_stream(), media_type=media_type)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_startup()
    app.state.engine.load()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Holos TTS", lifespan=lifespan)
    app.state.engine = SynthesisEngine(settings)
    app.include_router(router)
    return app


app = create_app()

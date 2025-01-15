import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.tts_request import TtsRequest
from src.models.wavefile_content import WavefileContent
from transformers import AutoProcessor, AutoModel

logger = logging.getLogger(__name__)
router = APIRouter()
MODEL_NAME = "suno/bark-small"


@router.post("/", response_model=WavefileContent, response_class=JSONResponse)
async def generate_speech(body: TtsRequest) -> WavefileContent:
    logger.info("Loading model %s", MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir="/model")
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir="/model")

    voice_preset = f"v2/{body.language}_speaker_2"
    inputs = processor(
        text=body.text,
        voice_preset=voice_preset,
        return_tensors="pt",
    )

    logger.info("Start generating speech using voice preset: %s", voice_preset)
    speech_values = model.generate(**inputs, do_sample=True)
    data = speech_values.cpu().numpy().squeeze()

    logger.info("Speech generation finished. Len: %s, dtype %s", len(data), data.dtype)

    return WavefileContent(
        rate=model.generation_config.sample_rate,
        data=data.tolist()
    )

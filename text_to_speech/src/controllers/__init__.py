from fastapi import APIRouter

from . import tts_controller

router = APIRouter()
router.include_router(tts_controller.router, prefix="/tts", tags=["text-to-speech"])

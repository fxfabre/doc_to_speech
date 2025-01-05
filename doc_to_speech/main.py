import logging
from pathlib import Path

import bark
import langid

from doc_to_speech.ocr.img_cleaning import clean_image
from doc_to_speech.ocr.img_processing import extract_text_from_image
from doc_to_speech.text_to_speech.controllers.tts_controller import (
    generate_speech_from_lib,
)

logger = logging.getLogger(__name__)


def get_audio_for_file(doc_file_path: Path):
    logger.info("Using CPU/GPU device : %s", bark.generation._grab_best_device())

    clean_img_path = clean_image(doc_file_path)
    text = extract_text_from_image(clean_img_path)
    language = langid.classify(text)[0]

    audio = generate_speech_from_lib(text, language)
    return audio

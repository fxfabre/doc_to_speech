import logging
import os
from pathlib import Path

import nltk
import numpy as np
import scipy
from bark import SAMPLE_RATE, generate_audio, preload_models
from pydantic.v1.validators import bool_validator
from sympy import false
from transformers import AutoProcessor, BarkModel

from doc_to_speech.common.decorators import timeit
from doc_to_speech.text_to_speech.models.wavefile_content import WavefileContent

logger = logging.getLogger(__name__)
VOICE_PRESETS = {
    "fr": "v2/fr_speaker_2",
    "en": "v2/en_speaker_6",
}


@timeit
def generate_speech_from_transformer(text: str, language: str) -> WavefileContent:
    model_name = _get_model_name()
    logger.info("Loading model %s", model_name)
    processor = AutoProcessor.from_pretrained(
        model_name, cache_dir=os.environ["HF_HOME"]
    )
    model = BarkModel.from_pretrained(model_name, cache_dir=os.environ["HF_HOME"])
    sample_rate = model.generation_config.sample_rate

    voice_preset = VOICE_PRESETS[language]
    sentences = nltk.sent_tokenize(text)
    silence = np.zeros(int(0.25 * sample_rate))
    pieces = []
    for sentence in sentences:
        inputs = processor(
            text=sentence,
            voice_preset=voice_preset,
            # return_tensors="pt",
        )

        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        pieces += [audio_array, silence.copy()]

    return WavefileContent(rate=sample_rate, data=np.concatenate(pieces))


@timeit
def generate_speech_from_lib(text: str, language: str) -> WavefileContent:
    voice_preset = VOICE_PRESETS[language]
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    # download and load all models
    preload_models(
        text_use_small=_use_small_model(),
        coarse_use_small=_use_small_model(),
        fine_use_small=_use_small_model(),
    )

    pieces = []
    for sentence in nltk.sent_tokenize(text):
        audio_array = generate_audio(sentence, history_prompt=voice_preset)
        pieces += [audio_array, silence.copy()]

    return WavefileContent(rate=SAMPLE_RATE, data=np.concatenate(pieces))


def save_audio(audio: WavefileContent, audio_path: Path):
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving audio file to %s", audio_path)

    scipy.io.wavfile.write(audio_path.as_posix(), rate=audio.rate, data=audio.data)


def _use_small_model():
    return bool_validator(os.getenv("SUNO_USE_SMALL_MODELS", false))


def _get_model_name() -> str:
    if _use_small_model():
        return "suno/bark-small"
    return "suno/bark"

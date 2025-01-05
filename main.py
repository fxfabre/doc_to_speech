import logging
from pathlib import Path

from doc_to_speech.main import get_audio_for_file
from doc_to_speech.text_to_speech.controllers.tts_controller import save_audio

logger = logging.getLogger(__name__)


def main():
    folder_articles = Path("tests/images")
    folder_audio_out = Path("tests/audio_out")

    for image_file_path in folder_articles.rglob("*.png"):
        audio_file_path = (folder_audio_out / str(image_file_path.stem)).with_suffix(
            ".wav"
        )

        logger.info(
            "Processing file %s, Saving audio to %s", image_file_path, audio_file_path
        )
        audio = get_audio_for_file(image_file_path)
        save_audio(audio, audio_file_path)


if __name__ == "__main__":
    main()

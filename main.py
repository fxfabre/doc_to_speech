import posixpath

import numpy as np
import os
import requests
import scipy
from pathlib import Path

API_BASE_URL = "http://192.168.1.90:5000/api/v1"


def main():
    folder_articles = Path("articles")
    folder_audio_out = Path("audio_out")

    for root, sub_dirs, files in os.walk(folder_articles):
        relative_dir = Path(root).relative_to(folder_articles)

        for file_name in files:
            text_file = os.path.join(root, file_name)
            audio_file = folder_audio_out / relative_dir / Path(file_name).stem

            print("Processing file", text_file)
            file_content = Path(text_file).read_text()[:50]
            call_api_tts_and_save(file_content, audio_file)


def call_api_tts_and_save(text: str, audio_file_name: str | Path) -> Path:
    body = {
        "language": "fr",
        "text": text
    }
    response = requests.post(posixpath.join(API_BASE_URL, "tts"), json=body)
    response.raise_for_status()
    json_content = response.json()

    audio_file_path = audio_file_name.with_name(audio_file_name.stem + ".wav")
    print("Saving audio file to", audio_file_path)

    audio_file_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(
        audio_file_path.open("wb"),
        rate=json_content["rate"],
        data=np.array(json_content["data"])
    )
    return audio_file_path


if __name__ == '__main__':
    main()

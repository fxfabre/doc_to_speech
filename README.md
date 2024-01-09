# doc_to_speech
Book reader : From word / pdf to audio

Chaque sous dossier est un projet / une API indépendante :
- tesseract_fast_api : en developpement
- text_to_speech : TTS avec suno / bark


## Exemple d'utilisation

Lancement de l'API :
- `docker compose up --build tts-api`

Generation de la voix :
```python
import requests
import scipy
import numpy as np

body = {
    "language": "fr",
    "text": "Ceci est un test de génération de parole."
}
response = requests.post("http://localhost:5000/api/v1/tts", json=body)

scipy.io.wavfile.write(
    "sample.wav",
    rate=response.json()["rate"],
    data=np.array(response.json()["data"])
)
```

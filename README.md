# doc_to_speech
Book reader : From word / pdf to audio

Un dossier par fonctionnalité :
- ocr : reconnaissance de caractères sur un pdf / une image
- text_to_speech : TTS avec suno / bark


## Text to speech
Sources :
- Bark model : https://huggingface.co/docs/transformers/model_doc/bark
- TTS model from facebook : https://huggingface.co/facebook/tts_transformer-fr-cv7_css10
- Packaged espeak API : https://github.com/parente/espeakbox

Configure Accelerate
- To optimize GPU usage : [accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/install)
- Run `accelerate config` & check config : `accelerate env`
- Config file at `./model/accelerate/default_config.yaml`


## Setup project
1. Create `.env` file with :
    ```
    SUNO_USE_SMALL_MODELS=true
    SUNO_ENABLE_MPS=true
    HF_HOME=./model_cache
    ```
2. Install `uv`
2. Install dependencies : `uv pip install -r requirements.txt`
3. Run : `python main.py`

Sources :
- Bark model : https://huggingface.co/docs/transformers/model_doc/bark
- TTS model from facebook : https://huggingface.co/facebook/tts_transformer-fr-cv7_css10
- Packaged espeak API : https://github.com/parente/espeakbox

Configure Accelerate
- To optimize GPU usage : [accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/install)
- Run `accelerate config` & check config : `accelerate env`
- Config file at `./model/accelerate/default_config.yaml`

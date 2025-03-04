import logging.config

import dotenv

# logging.config.fileConfig("config/logging.ini")

logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

dotenv.load_dotenv()

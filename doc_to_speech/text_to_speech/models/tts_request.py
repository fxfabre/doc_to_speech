import logging
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TtsRequest(BaseModel):
    text: str
    language: Literal["fr", "en"]

import logging

from pydantic import BaseModel
from typing import Literal

logger = logging.getLogger(__name__)


class TtsRequest(BaseModel):
    text: str
    language: Literal["fr", "en"]

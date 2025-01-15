import logging

from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)


class WavefileContent(BaseModel):
    rate: int
    data: List[float]

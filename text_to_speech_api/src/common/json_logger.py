from typing import Any
from typing import Dict

from pythonjsonlogger.jsonlogger import JsonFormatter


class CustomJsonFormatter(JsonFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, rename_fields=self._rename_fields, **kwargs)

    @property
    def _rename_fields(self) -> Dict[str, str]:
        return {
            "name": "source",
            "levelname": "severity",
        }

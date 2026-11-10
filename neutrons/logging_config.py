from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from neutrons.config import settings


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict) and context:
            entry["context"] = context
        if record.exc_info:
            entry["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)
    for handler in root_logger.handlers:
        handler.setFormatter(JsonLogFormatter())

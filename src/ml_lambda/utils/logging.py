"""Sistema de logging estructurado."""

import json
import logging
from datetime import datetime
from typing import Any


class StructuredLogger:
    """Logger con salida JSON estructurada."""

    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level))

        # Configurar handler si no existe
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def _format_message(self, level: str, message: str, **kwargs: Any) -> str:
        """Formatea mensaje como JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs,
        }
        return json.dumps(log_entry)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log nivel INFO."""
        self._logger.info(self._format_message("INFO", message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log nivel ERROR."""
        self._logger.error(self._format_message("ERROR", message, **kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log nivel DEBUG."""
        self._logger.debug(self._format_message("DEBUG", message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log nivel WARNING."""
        self._logger.warning(self._format_message("WARNING", message, **kwargs))

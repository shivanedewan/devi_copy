from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_FORMAT = "%(asctime)s - %(levelname)s - [ %(filename)s:%(lineno)d | %(funcName)s() ] %(message)s"
_DEFAULT_LEVEL = "INFO"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def _resolve_level() -> int:
    level_name = str(os.getenv("CONTEXT_FETCHER_LOG_LEVEL", _DEFAULT_LEVEL)).upper().strip()
    return getattr(logging, level_name, logging.INFO)


def _has_file_handler(logger: logging.Logger, log_file: Path) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            base = getattr(handler, "baseFilename", "")
            if base and Path(base).resolve() == log_file.resolve():
                return True
    return False


def setup_logging() -> None:
    level = _resolve_level()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ContextFetcherProd.log"

    formatter = logging.Formatter(_LOG_FORMAT)

    if not _has_file_handler(root_logger, log_file):
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Keep uvicorn logger levels aligned; do not remove existing handlers.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).setLevel(level)


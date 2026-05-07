from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from prompt_defaults import DEFAULT_RAG_SYSTEM_PROMPT


_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_PROMPT_DIR = _THIS_DIR / "prompts" / "system" / "default"
_PROMPT_FILENAME = "rag_system_prompt.md"
PROMPT_LABEL = "RAG System Prompt"
logger = logging.getLogger(__name__)


# changed
def _normalize_prompt_text(content: str) -> str:
    text = str(content or "").strip()
    return f"{text}\n" if text else ""


def get_default_prompt_text() -> str:
    return _normalize_prompt_text(DEFAULT_RAG_SYSTEM_PROMPT)


def _shared_prompt_dir() -> str:
    env_value = str(os.getenv("LLM_SHARED_PROMPT_DIR") or "").strip()
    if env_value:
        return env_value
    try:
        from settings import settings
    except Exception as exc:
        logger.warning("Failed to load settings for shared prompt directory; using local prompt file: %s", exc)
        return ""
    return str(getattr(settings, "llm_shared_prompt_dir", "") or "").strip()


def get_prompt_file_path(prompt_file: Optional[Path | str] = None) -> Path:
    if prompt_file is not None:
        return Path(prompt_file)
    shared_dir = _shared_prompt_dir()
    if shared_dir:
        return Path(shared_dir).expanduser() / _PROMPT_FILENAME
    return _LOCAL_PROMPT_DIR / _PROMPT_FILENAME


def ensure_prompt_file(prompt_file: Optional[Path | str] = None) -> Path:
    path = get_prompt_file_path(prompt_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(get_default_prompt_text(), encoding="utf-8")
    except Exception as exc:
        logger.warning(
            "Failed to ensure RAG prompt file path=%s; bundled default will be used if read fails: %s",
            path,
            exc,
        )
    return path


def read_active_prompt(prompt_file: Optional[Path | str] = None) -> str:
    path = ensure_prompt_file(prompt_file)
    try:
        current_text = _normalize_prompt_text(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read active RAG prompt path=%s; using bundled default: %s", path, exc)
        return get_default_prompt_text()
    return current_text or get_default_prompt_text()


def write_active_prompt(content: str, prompt_file: Optional[Path | str] = None) -> Path:
    normalized_text = _normalize_prompt_text(content)
    if not normalized_text:
        raise ValueError("Prompt content cannot be empty.")

    path = get_prompt_file_path(prompt_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized_text, encoding="utf-8")
    return path


def reset_active_prompt(prompt_file: Optional[Path | str] = None) -> str:
    default_text = get_default_prompt_text()
    write_active_prompt(default_text, prompt_file)
    return default_text


def get_prompt_metadata(prompt_file: Optional[Path | str] = None) -> Dict[str, Any]:
    path = ensure_prompt_file(prompt_file)
    return {
        "label": PROMPT_LABEL,
        "filename": path.name,
        "path": str(path),
    }

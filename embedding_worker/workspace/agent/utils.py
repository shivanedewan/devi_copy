import os
import json
from pathlib import Path
import shutil
from transformers import AutoTokenizer

from settings import settings

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(r"./tokenizers/Qwen8B")

def chunk_text(text, chunk_size=settings.chunk_size, overlap=50):
    tokens = tokenizer.encode(text)
    print(f"Number of tokens found: {len(tokens)}")
    chunks = []
    
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start += chunk_size - overlap
    print(f"Number of chunks: {len(chunks)}")
    return chunks
    
def clean_tmp(tmp_dir: str | Path, *, remove_subdirs: bool = False) -> None:
    tmp_path = Path(tmp_dir)

    if not tmp_path.is_dir():
        raise NotADirectoryError(f"{tmp_path!s} is not a directory")

    for entry in tmp_path.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
            elif entry.is_dir() and remove_subdirs:
                shutil.rmtree(entry)
        except Exception as exc:
            # log or ignore according to your needs
            print(f"Failed to delete {entry!s}: {exc}")


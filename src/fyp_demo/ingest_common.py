"""Helpers shared by both ingestion pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def iter_source_files(data_dir: Path) -> Iterator[Path]:
    """Yield source files in ``data_dir`` with a supported extension, sorted by name."""
    if not data_dir.exists():
        return
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def save_raw_text(out_dir: Path, stem: str, text: str) -> Path:
    """Write ``text`` to ``out_dir/<stem>.txt``, creating the directory if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path

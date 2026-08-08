"""Shared paths, model names, and pipeline defaults for the fyp_demo comparison app."""

from __future__ import annotations

from pathlib import Path

from akshara_kit.contracts.chunking import ChunkConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
RAW_TEXT_AKSHARA_DIR = STORAGE_DIR / "raw_text" / "akshara"
RAW_TEXT_BASELINE_DIR = STORAGE_DIR / "raw_text" / "baseline"

AKSHARA_COLLECTION_NAME = "akshara_pipeline"
BASELINE_COLLECTION_NAME = "langchain_baseline"

EMBEDDING_MODEL_NAME = "Nimsara2001/labse-sinhala-finetuned"

OPENAI_MODEL = "gpt-4o-mini"

# Matches akshara-kit's own ChunkConfig.max_words default so the baseline's
# TokenTextSplitter chunk_size can be set to the same value — the comparison
# is about boundary quality, not average chunk length.
CHUNK_CONFIG = ChunkConfig()

BASELINE_CHUNK_SIZE = CHUNK_CONFIG.max_words
BASELINE_CHUNK_OVERLAP = 50

DEFAULT_RETRIEVAL_K = 4

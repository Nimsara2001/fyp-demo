"""Pipeline A: akshara-kit's Sinhala-aware extraction + neuro-symbolic chunking."""

from __future__ import annotations

import logging
from pathlib import Path

from akshara_kit import (
    ExtractionFailedError,
    PrologUnavailableError,
    UnsupportedFormatError,
    route,
)
from akshara_kit.brain import HybridChunker
from akshara_kit.contracts.chunking import ChunkedDocument, SemanticChunk
from akshara_kit.contracts.extraction import ExtractionResult

from fyp_demo import config
from fyp_demo.embeddings import SinhalaLabseEmbeddings, build_shared_labse_scorer, get_shared_encoder
from fyp_demo.ingest_common import iter_source_files, save_raw_text
from fyp_demo.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def _save_raw_extraction(path: Path, result: ExtractionResult) -> None:
    """Dump raw text + metadata for the UI's "peek at raw extraction" panel.

    ``model_dump_json(exclude={"text"})`` (rather than hand-picking fields) serialises
    the ``FontDetectionMethod`` enum correctly and covers ``quality`` being ``None`` on
    a garbled/empty extraction, so the panel can render "quality: n/a" instead of
    crashing.
    """
    save_raw_text(config.RAW_TEXT_AKSHARA_DIR, path.stem, result.text)
    meta_path = config.RAW_TEXT_AKSHARA_DIR / f"{path.stem}.meta.json"
    meta_path.write_text(result.model_dump_json(indent=2, exclude={"text"}), encoding="utf-8")


def _chunk_metadata(chunk: SemanticChunk) -> dict:
    """Flatten a ``SemanticChunk`` into the scalar-only shape Chroma metadata requires."""
    meta = {
        "chunk_id": chunk.chunk_id,
        "index": chunk.index,
        "word_count": chunk.word_count,
        "source_document": chunk.source_document or "",
        "source_format": chunk.source_format.value if chunk.source_format else "",
        "segment_kind": chunk.segment_kind.value,
        "boundaries": ",".join(b.value for b in chunk.boundaries),
        "merged_from": chunk.merged_from,
        "pipeline": "akshara",
    }
    if chunk.quality is not None:
        meta["sinhala_ratio"] = chunk.quality.sinhala_ratio
    return meta


def ingest_akshara(data_dir: Path = config.DATA_RAW_DIR, *, rebuild: bool = False) -> int:
    """Ingest every supported file in ``data_dir`` into the akshara-kit Chroma collection.

    Opens one ``HybridChunker`` (one SWI-Prolog subprocess) for the whole batch instead
    of calling the module-level ``chunk()`` per file. Returns total chunks added.
    """
    encoder = get_shared_encoder()
    embedder = SinhalaLabseEmbeddings(encoder)
    scorer = build_shared_labse_scorer(encoder)
    vs = get_vectorstore(config.AKSHARA_COLLECTION_NAME, embedder, rebuild=rebuild)

    total_chunks = 0
    with HybridChunker(config=config.CHUNK_CONFIG, scorer=scorer) as chunker:
        for path in iter_source_files(data_dir):
            try:
                result = route(str(path))
            except (ExtractionFailedError, UnsupportedFormatError):
                logger.exception("Failed to extract %s with akshara-kit", path.name)
                continue

            _save_raw_extraction(path, result)

            try:
                doc: ChunkedDocument = chunker.chunk(result, source_document=path.name)
            except PrologUnavailableError:
                logger.exception("SWI-Prolog unavailable while chunking %s", path.name)
                continue

            if not doc:
                logger.warning("No chunks produced for %s (akshara)", path.name)
                continue

            ids = [c.chunk_id for c in doc]
            metadatas = [_chunk_metadata(c) for c in doc]
            vs.add_texts(texts=doc.texts, metadatas=metadatas, ids=ids)
            total_chunks += len(doc)
            logger.info("Ingested %d akshara-kit chunks from %s", len(doc), path.name)

    return total_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = ingest_akshara()
    print(f"Ingested {n} akshara-kit chunks")

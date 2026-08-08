"""Pipeline B: a deliberately naive LangChain baseline (generic loaders + fixed-token
splitting), so the demo shows what a generic RAG pipeline does to the same documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

from fyp_demo import config
from fyp_demo.embeddings import STEmbeddings, get_baseline_encoder
from fyp_demo.ingest_common import iter_source_files, save_raw_text
from fyp_demo.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}


def _load_raw_text(path: Path) -> str | None:
    loader_cls = LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        logger.warning("No baseline loader for %s, skipping", path.name)
        return None
    docs = loader_cls(str(path)).load()
    return "\n\n".join(d.page_content for d in docs)


def ingest_baseline(data_dir: Path = config.DATA_RAW_DIR, *, rebuild: bool = False) -> int:
    """Ingest every supported file in ``data_dir`` into the baseline Chroma collection.

    Returns the total number of chunks added.
    """
    encoder = get_baseline_encoder()
    embedder = STEmbeddings(encoder)
    vs = get_vectorstore(config.BASELINE_COLLECTION_NAME, embedder, rebuild=rebuild)

    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=config.BASELINE_CHUNK_SIZE,
        chunk_overlap=config.BASELINE_CHUNK_OVERLAP,
    )

    total_chunks = 0
    for path in iter_source_files(data_dir):
        try:
            raw_text = _load_raw_text(path)
        except Exception:
            logger.exception("Failed to load %s for baseline pipeline", path.name)
            continue
        if raw_text is None:
            continue

        save_raw_text(config.RAW_TEXT_BASELINE_DIR, path.stem, raw_text)

        chunks = splitter.split_text(raw_text)
        if not chunks:
            logger.warning("No chunks produced for %s (baseline)", path.name)
            continue

        ids = [f"{path.stem}-baseline-{i}" for i in range(len(chunks))]
        metadatas = [
            {"source_document": path.name, "pipeline": "baseline", "chunk_index": i}
            for i in range(len(chunks))
        ]
        vs.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
        total_chunks += len(chunks)
        logger.info("Ingested %d baseline chunks from %s", len(chunks), path.name)

    return total_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = ingest_baseline()
    print(f"Ingested {n} baseline chunks")

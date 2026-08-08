"""Chroma vector store helpers shared by both ingestion pipelines and the app."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from fyp_demo import config


def get_vectorstore(collection_name: str, embedder: Embeddings, *, rebuild: bool = False) -> Chroma:
    """Open the persistent Chroma collection ``collection_name``.

    Both pipelines share the one ``storage/chroma`` directory but live in separate
    named collections. Pass ``rebuild=True`` to drop and recreate the collection first
    (used by the ``--rebuild`` CLI flag).
    """
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=str(config.CHROMA_DIR),
    )
    if rebuild:
        reset_collection(vs)
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=str(config.CHROMA_DIR),
        )
    return vs


def reset_collection(vs: Chroma) -> None:
    vs.delete_collection()

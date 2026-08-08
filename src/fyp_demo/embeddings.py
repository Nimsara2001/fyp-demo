"""Embedding functions for both pipelines.

akshara-kit pipeline: the project's fine-tuned Sinhala LaBSE (``get_shared_encoder``),
also reused for akshara-kit's own chunk-merge coherence scoring so the ~1.8GB checkpoint
loads once per process.

Baseline pipeline: a separate, generic, non-Sinhala-tuned model
(``get_baseline_encoder``), by deliberate choice — see the note on
``config.BASELINE_EMBEDDING_MODEL_NAME``. This means the two pipelines no longer differ
only in extraction/chunking; embeddings are a second variable now.
"""

from __future__ import annotations

from functools import lru_cache

from akshara_kit.brain.encoder import LabseScorer
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from fyp_demo import config


@lru_cache(maxsize=1)
def get_shared_encoder() -> SentenceTransformer:
    """akshara-kit pipeline's encoder: fine-tuned Sinhala LaBSE, loaded once per process."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_baseline_encoder() -> SentenceTransformer:
    """Baseline pipeline's encoder: generic, non-Sinhala-tuned, loaded once per process."""
    return SentenceTransformer(config.BASELINE_EMBEDDING_MODEL_NAME)


class STEmbeddings(Embeddings):
    """LangChain ``Embeddings`` wrapper around any sentence-transformers encoder."""

    def __init__(self, encoder: SentenceTransformer):
        self._encoder = encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encoder.encode(text, normalize_embeddings=True).tolist()


def build_shared_labse_scorer(encoder: SentenceTransformer) -> LabseScorer:
    """Build a ``LabseScorer`` that reuses ``encoder`` instead of loading its own copy.

    ``LabseScorer`` lazy-loads into ``self._model`` on first ``.embed()``/``.score()``
    call, guarded by ``if self._model is not None: return self._model``. Setting the
    private attribute before that first call means the lazy-load branch never runs, so
    the coherence scorer used by akshara-kit's chunk merge shares the exact same model
    instance as the akshara-kit pipeline's Chroma embedding function.

    This relies on a private attribute, not public API — guarded with ``hasattr`` so a
    future akshara-kit release that removes ``_model`` just falls back to the scorer
    loading its own copy instead of raising.
    """
    scorer = LabseScorer(model_name=config.EMBEDDING_MODEL_NAME)
    if hasattr(scorer, "_model"):
        scorer._model = encoder
    return scorer

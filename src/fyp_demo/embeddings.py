"""Shared Sinhala LaBSE encoder, reused for both Chroma embedding and akshara-kit's
chunk-merge coherence scoring so the ~1.8GB checkpoint is loaded once per process.
"""

from __future__ import annotations

from functools import lru_cache

from akshara_kit.brain.encoder import LabseScorer
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from fyp_demo import config


@lru_cache(maxsize=1)
def get_shared_encoder() -> SentenceTransformer:
    """Process-wide singleton so the checkpoint is loaded exactly once."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


class SinhalaLabseEmbeddings(Embeddings):
    """LangChain ``Embeddings`` wrapper around the shared fine-tuned LaBSE encoder."""

    def __init__(self, encoder: SentenceTransformer | None = None):
        self._encoder = encoder or get_shared_encoder()

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
    instance as Chroma's embedding function.

    This relies on a private attribute, not public API — guarded with ``hasattr`` so a
    future akshara-kit release that removes ``_model`` just falls back to the scorer
    loading its own copy instead of raising.
    """
    scorer = LabseScorer(model_name=config.EMBEDDING_MODEL_NAME)
    if hasattr(scorer, "_model"):
        scorer._model = encoder
    return scorer

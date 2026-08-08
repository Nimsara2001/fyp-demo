"""End-to-end smoke test: run one canned question through both pipelines.

Catches prompt/API-key/model issues before the Streamlit UI exists. Run after both
ingest_baseline and ingest_akshara have populated their collections.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from fyp_demo import config
from fyp_demo.embeddings import STEmbeddings, get_baseline_encoder, get_shared_encoder
from fyp_demo.rag import RagPipeline, build_llm
from fyp_demo.vectorstore import get_vectorstore

QUESTION = "ලේඛනයෙන් පෙළ උපුටා ගැනීමෙන් පසු කරන පියවර මොනවාද?"


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)

    akshara_embedder = STEmbeddings(get_shared_encoder())
    baseline_embedder = STEmbeddings(get_baseline_encoder())
    llm = build_llm()

    akshara_vs = get_vectorstore(config.AKSHARA_COLLECTION_NAME, akshara_embedder)
    baseline_vs = get_vectorstore(config.BASELINE_COLLECTION_NAME, baseline_embedder)

    akshara_pipeline = RagPipeline(akshara_vs, "akshara-kit", llm, k=config.DEFAULT_RETRIEVAL_K)
    baseline_pipeline = RagPipeline(baseline_vs, "baseline", llm, k=config.DEFAULT_RETRIEVAL_K)

    print(f"Question: {QUESTION}\n")
    for pipeline in (akshara_pipeline, baseline_pipeline):
        result = pipeline.answer(QUESTION)
        print(f"=== {result.pipeline_name} ===")
        print(f"Answer: {result.answer}")
        print(
            f"Retrieval: {result.retrieval_latency_s:.2f}s  "
            f"Generation: {result.generation_latency_s:.2f}s"
        )
        print(f"Retrieved {len(result.retrieved)} chunks")
        print()


if __name__ == "__main__":
    main()

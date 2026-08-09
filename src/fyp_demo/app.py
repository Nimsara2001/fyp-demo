"""Streamlit demo: akshara-kit pipeline vs. a naive LangChain baseline, side by side."""

from __future__ import annotations

import concurrent.futures
import json

import streamlit as st
from dotenv import load_dotenv

from fyp_demo import config
from fyp_demo.embeddings import STEmbeddings, get_baseline_encoder, get_shared_encoder
from fyp_demo.rag import PipelineResult, RagPipeline, build_llm
from fyp_demo.vectorstore import get_vectorstore

load_dotenv()

st.set_page_config(page_title="akshara-kit vs. baseline RAG", layout="wide")


@st.cache_resource
def get_app_resources():
    """Build the expensive shared resources exactly once per server process.

    Each collection must be queried with the same embedding model it was ingested
    with (akshara-kit: fine-tuned Sinhala LaBSE; baseline: a separate, generic
    model — see ``config.BASELINE_EMBEDDING_MODEL_NAME``), so this builds two
    distinct embedders rather than sharing one.

    Deliberately excludes the LLM: ``ChatOpenAI`` validates credentials in its
    constructor, so building it here would crash the whole page (sidebar, raw-text
    peek panel, everything) on a missing API key instead of only the chat turn that
    actually needs it.
    """
    akshara_embedder = STEmbeddings(get_shared_encoder())
    baseline_embedder = STEmbeddings(get_baseline_encoder())
    akshara_vs = get_vectorstore(config.AKSHARA_COLLECTION_NAME, akshara_embedder)
    baseline_vs = get_vectorstore(config.BASELINE_COLLECTION_NAME, baseline_embedder)
    return akshara_vs, baseline_vs


@st.cache_resource
def get_llm():
    return build_llm()


def collection_stats(vs) -> tuple[int, int]:
    """(chunk count, distinct source-document count) for a collection."""
    data = vs.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    docs = {m.get("source_document") for m in metadatas if m and m.get("source_document")}
    return len(metadatas), len(docs)


def chunks_for_document(vs, source_document: str) -> int:
    return len(vs.get(where={"source_document": source_document})["ids"])


EXTENSION_BY_SOURCE_FORMAT = {"pdf": ".pdf", "docx": ".docx", "xlsx": ".xlsx"}


def run_both(
    akshara_pipeline: RagPipeline,
    baseline_pipeline: RagPipeline,
    query: str,
    source_document: str,
):
    doc_filter = {"source_document": source_document}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        akshara_future = pool.submit(akshara_pipeline.answer, query, filter=doc_filter)
        baseline_future = pool.submit(baseline_pipeline.answer, query, filter=doc_filter)
        return akshara_future.result(), baseline_future.result()


def render_result(column, title: str, result: PipelineResult):
    with column:
        st.subheader(title)
        st.write(result.answer)
        with st.expander(f"Retrieved chunks ({len(result.retrieved)})"):
            for i, item in enumerate(result.retrieved, start=1):
                st.markdown(f"**Chunk {i}** — score: {item['score']:.3f}")
                st.text(item["text"])
                st.json(item["metadata"], expanded=False)
        st.caption(
            f"retrieval: {result.retrieval_latency_s:.2f}s · "
            f"generation: {result.generation_latency_s:.2f}s"
        )


def get_document_stems() -> list[str]:
    meta_files = sorted(config.RAW_TEXT_AKSHARA_DIR.glob("*.meta.json"))
    return [f.name.removesuffix(".meta.json") for f in meta_files]


def load_akshara_meta(stem: str) -> dict:
    return json.loads((config.RAW_TEXT_AKSHARA_DIR / f"{stem}.meta.json").read_text(encoding="utf-8"))


def source_document_name(stem: str, meta: dict) -> str:
    extension = EXTENSION_BY_SOURCE_FORMAT.get(meta.get("source_format"), "")
    return f"{stem}{extension}"


def render_document_summary(akshara_vs, baseline_vs, meta: dict, source_document: str):
    """Chunk counts and font-detection evidence for the selected document — no raw chunk dump."""
    akshara_chunks = chunks_for_document(akshara_vs, source_document)
    baseline_chunks = chunks_for_document(baseline_vs, source_document)

    quality = meta.get("quality") or {}
    sinhala_ratio = quality.get("sinhala_ratio")
    legacy_fonts = meta.get("detected_legacy_fonts") or []

    with st.expander(f"Document summary — {source_document}", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**akshara-kit pipeline**")
            st.metric("Chunks", akshara_chunks)
            st.write(f"Backend: `{meta.get('backend_id', 'n/a')}`")
            st.write(f"Font detection method: `{meta.get('font_detection_method', 'none')}`")
            st.write("Legacy fonts detected: " + (", ".join(f"`{f}`" for f in legacy_fonts) if legacy_fonts else "none"))
            st.write(f"OCR used: {meta.get('ocr_used', False)}")
            st.write(f"Sinhala ratio: {sinhala_ratio:.3f}" if sinhala_ratio is not None else "Sinhala ratio: n/a")
        with col_b:
            st.markdown("**LangChain baseline**")
            st.metric("Chunks", baseline_chunks)
            st.write("Font detection: not supported (generic loader)")


def main():
    akshara_vs, baseline_vs = get_app_resources()

    st.title("Sinhala RAG: akshara-kit vs. a naive LangChain baseline")

    stems = get_document_stems()
    if not stems:
        st.info("No ingested documents yet — run ingestion first.")
        return

    with st.sidebar:
        st.header("Collections")
        akshara_chunks, akshara_docs = collection_stats(akshara_vs)
        baseline_chunks, baseline_docs = collection_stats(baseline_vs)
        st.metric("akshara-kit chunks", akshara_chunks, help=f"{akshara_docs} documents")
        st.metric("baseline chunks", baseline_chunks, help=f"{baseline_docs} documents")
        k = st.slider("Retrieved chunks (k)", min_value=1, max_value=25, value=config.DEFAULT_RETRIEVAL_K)

        st.header("Document")
        selected_stem = st.selectbox("Query this document", stems, key="selected_document")

    meta = load_akshara_meta(selected_stem)
    source_document = source_document_name(selected_stem, meta)

    # A document switch starts a fresh conversation — mixing turns scoped to different
    # documents in one thread would be confusing, and retrieval for the older turns
    # would silently stop matching what's now selected.
    if st.session_state.get("history_document") != source_document:
        st.session_state.history = []
        st.session_state.history_document = source_document

    render_document_summary(akshara_vs, baseline_vs, meta, source_document)
    st.caption(f"Chat is scoped to: **{source_document}**")

    for query, akshara_result, baseline_result in st.session_state.history:
        st.chat_message("user").write(query)
        col1, col2 = st.columns(2)
        render_result(col1, "akshara-kit pipeline", akshara_result)
        render_result(col2, "LangChain baseline", baseline_result)

    query = st.chat_input(f"Ask a question about {source_document} (Sinhala)")
    if query:
        try:
            llm = get_llm()
        except Exception as exc:
            st.error(f"Could not create the OpenAI client — check OPENAI_API_KEY in .env. ({exc})")
            st.stop()

        akshara_pipeline = RagPipeline(akshara_vs, "akshara-kit", llm, k=k)
        baseline_pipeline = RagPipeline(baseline_vs, "baseline", llm, k=k)
        with st.spinner("Running both pipelines..."):
            akshara_result, baseline_result = run_both(
                akshara_pipeline, baseline_pipeline, query, source_document
            )
        st.session_state.history.append((query, akshara_result, baseline_result))
        st.rerun()


if __name__ == "__main__":
    main()

"""Streamlit demo: akshara-kit pipeline vs. a naive LangChain baseline, side by side."""

from __future__ import annotations

import concurrent.futures

import streamlit as st
from dotenv import load_dotenv

from fyp_demo import config
from fyp_demo.embeddings import SinhalaLabseEmbeddings, get_shared_encoder
from fyp_demo.rag import PipelineResult, RagPipeline, build_llm
from fyp_demo.vectorstore import get_vectorstore

load_dotenv()

st.set_page_config(page_title="akshara-kit vs. baseline RAG", layout="wide")


@st.cache_resource
def get_app_resources():
    """Build the expensive shared resources exactly once per server process.

    Deliberately excludes the LLM: ``ChatOpenAI`` validates credentials in its
    constructor, so building it here would crash the whole page (sidebar, raw-text
    peek panel, everything) on a missing API key instead of only the chat turn that
    actually needs it.
    """
    encoder = get_shared_encoder()
    embedder = SinhalaLabseEmbeddings(encoder)
    akshara_vs = get_vectorstore(config.AKSHARA_COLLECTION_NAME, embedder)
    baseline_vs = get_vectorstore(config.BASELINE_COLLECTION_NAME, embedder)
    return embedder, akshara_vs, baseline_vs


@st.cache_resource
def get_llm():
    return build_llm()


def collection_stats(vs) -> tuple[int, int]:
    """(chunk count, distinct source-document count) for a collection."""
    data = vs.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    docs = {m.get("source_document") for m in metadatas if m and m.get("source_document")}
    return len(metadatas), len(docs)


def run_both(akshara_pipeline: RagPipeline, baseline_pipeline: RagPipeline, query: str):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        akshara_future = pool.submit(akshara_pipeline.answer, query)
        baseline_future = pool.submit(baseline_pipeline.answer, query)
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


def render_raw_extraction_peek():
    akshara_files = sorted(config.RAW_TEXT_AKSHARA_DIR.glob("*.txt"))
    if not akshara_files:
        st.info("No ingested documents yet — run ingestion first.")
        return

    stems = [f.stem for f in akshara_files]
    with st.expander("Peek at raw extraction (akshara-kit vs. baseline)", expanded=False):
        selected = st.selectbox("Document", stems, key="peek_stem")

        akshara_text_path = config.RAW_TEXT_AKSHARA_DIR / f"{selected}.txt"
        baseline_text_path = config.RAW_TEXT_BASELINE_DIR / f"{selected}.txt"

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**akshara-kit extraction**")
            st.text(akshara_text_path.read_text(encoding="utf-8") if akshara_text_path.exists() else "(not ingested)")
        with col_b:
            st.markdown("**LangChain baseline extraction**")
            st.text(baseline_text_path.read_text(encoding="utf-8") if baseline_text_path.exists() else "(not ingested)")


def main():
    embedder, akshara_vs, baseline_vs = get_app_resources()

    st.title("Sinhala RAG: akshara-kit vs. a naive LangChain baseline")

    with st.sidebar:
        st.header("Collections")
        akshara_chunks, akshara_docs = collection_stats(akshara_vs)
        baseline_chunks, baseline_docs = collection_stats(baseline_vs)
        st.metric("akshara-kit chunks", akshara_chunks, help=f"{akshara_docs} documents")
        st.metric("baseline chunks", baseline_chunks, help=f"{baseline_docs} documents")
        k = st.slider("Retrieved chunks (k)", min_value=1, max_value=10, value=config.DEFAULT_RETRIEVAL_K)

    render_raw_extraction_peek()

    if "history" not in st.session_state:
        st.session_state.history = []

    for query, akshara_result, baseline_result in st.session_state.history:
        st.chat_message("user").write(query)
        col1, col2 = st.columns(2)
        render_result(col1, "akshara-kit pipeline", akshara_result)
        render_result(col2, "LangChain baseline", baseline_result)

    query = st.chat_input("Ask a question about the ingested documents (Sinhala)")
    if query:
        try:
            llm = get_llm()
        except Exception as exc:
            st.error(f"Could not create the OpenAI client — check OPENAI_API_KEY in .env. ({exc})")
            st.stop()

        akshara_pipeline = RagPipeline(akshara_vs, "akshara-kit", llm, k=k)
        baseline_pipeline = RagPipeline(baseline_vs, "baseline", llm, k=k)
        with st.spinner("Running both pipelines..."):
            akshara_result, baseline_result = run_both(akshara_pipeline, baseline_pipeline, query)
        st.session_state.history.append((query, akshara_result, baseline_result))
        st.rerun()


if __name__ == "__main__":
    main()

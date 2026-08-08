"""Shared RAG query/generation logic, reused for both pipelines.

Both pipeline instances use this exact same class, prompt template, and LLM instance
(``temperature=0``) — the only difference in output traces back to which Chroma
collection backs the retrieval, i.e. purely extraction+chunking quality.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from fyp_demo import config

SYSTEM_PROMPT = (
    "ඔබ ලබා දී ඇති සන්දර්භය (context) පමණක් පදනම් කරගෙන සිංහල භාෂාවෙන් පිළිතුරු දෙන "
    "සහායකයෙකි. සන්දර්භයේ පිළිතුර නොමැති නම්, 'මට ලබා දී ඇති තොරතුරු මත පදනම්ව මෙම "
    "ප්‍රශ්නයට පිළිතුරු දිය නොහැක.' යනුවෙන් පිළිතුරු දෙන්න. සන්දර්භයෙන් පිටත දැනුම භාවිත නොකරන්න."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "සන්දර්භය:\n{context}\n\nප්‍රශ්නය: {question}"),
    ]
)


@dataclass
class PipelineResult:
    pipeline_name: str
    answer: str
    retrieved: list[dict]
    retrieval_latency_s: float
    generation_latency_s: float


class RagPipeline:
    def __init__(self, vectorstore: VectorStore, label: str, llm: ChatOpenAI, k: int = 4):
        self.vectorstore = vectorstore
        self.label = label
        self.llm = llm
        self.k = k
        self._chain = PROMPT | self.llm

    def answer(self, query: str, *, filter: dict | None = None) -> PipelineResult:
        """``filter`` scopes retrieval to a metadata match (e.g. one source document)."""
        t0 = time.perf_counter()
        hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k, filter=filter)
        retrieval_latency = time.perf_counter() - t0

        retrieved = [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in hits
        ]
        context = "\n\n---\n\n".join(item["text"] for item in retrieved)

        t1 = time.perf_counter()
        response = self._chain.invoke({"context": context, "question": query})
        generation_latency = time.perf_counter() - t1

        return PipelineResult(
            pipeline_name=self.label,
            answer=response.content,
            retrieved=retrieved,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
        )


def build_llm(model: str | None = None) -> ChatOpenAI:
    resolved_model = model or os.environ.get("OPENAI_MODEL") or config.OPENAI_MODEL
    return ChatOpenAI(model=resolved_model, temperature=0)

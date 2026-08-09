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

REFUSAL = "මට ලබා දී ඇති තොරතුරු මත පදනම්ව මෙම ප්‍රශ්නයට පිළිතුරු දිය නොහැක."

# Instructions are in English deliberately: GPT models follow multi-clause conditional
# rules more reliably in their dominant training language, even when the required
# output language (Sinhala, enforced below) is different. Only the user-facing refusal
# sentence itself is Sinhala text, not an instruction.
SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly using the provided context, "
    "and you always respond in Sinhala.\n\n"
    "The context is made up of several segments separated by '---'. Each segment was "
    "retrieved independently. Judge EACH segment on its own — do not let a low-quality "
    "or corrupted segment lower your confidence in a different, clear segment sitting "
    "right next to it in the same context. It is normal for some segments to be badly "
    "corrupted while others in the very same context are clean and directly answer the "
    "question. If even one segment clearly and reliably answers the question, use it "
    "and disregard the corrupted segments.\n\n"
    "Segments often carry minor character-level corruption from PDF text extraction — "
    "e.g. Sinhala 'පොලී' appearing as 'පපොලී', 'යටතේ' appearing as 'යටපේ', extra 'ස්' "
    "characters, or swapped vowel signs. This is normal and expected. Do not discard a "
    "segment just because it has scattered corruption like this — use your knowledge of "
    "Sinhala to read past small letter-level glitches and extract the real words and "
    "facts. Numbers, amounts, dates, and URLs are not affected by this kind of "
    "corruption, so trust them.\n\n"
    "You may combine information stated across multiple segments into one complete "
    "answer — the full answer does not need to be in a single segment. But never add "
    "anything not present in the context, and never fill gaps using your own general "
    "knowledge — only reorganize, combine, and language-correct what the context "
    "actually states.\n\n"
    f"Only refuse — respond with exactly this fixed Sinhala sentence and nothing else: "
    f"'{REFUSAL}' — in these two cases:\n"
    "1. None of the segments contain any text recognizable as Sinhala or any other "
    "readable language at all (e.g. every single segment is unconverted legacy-font "
    "byte garbage, with no readable words anywhere in the whole context).\n"
    "2. Even after considering every segment individually, none of them contain any "
    "information relevant to the question.\n\n"
    "When in doubt, prefer attempting an answer from whatever clear, relevant segment "
    "you can find over refusing. Refusal is the last resort, not the default. Always "
    "write your final answer in Sinhala."
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

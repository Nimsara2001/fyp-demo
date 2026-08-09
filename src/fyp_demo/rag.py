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

SYSTEM_PROMPT = (
    "ඔබ ලබා දී ඇති සන්දර්භය (context) පමණක් පදනම් කරගෙන සිංහල භාෂාවෙන් පිළිතුරු දෙන සහායකයෙකි.\n\n"
    "සන්දර්භයේ පෙළ බොහෝ විට PDF නිස්සාරණයේදී ඇති වූ අකුරු මට්ටමේ දෝෂ සහිත ය — උදාහරණයක් ලෙස "
    "'පොලී' වෙනුවට 'පපොලී', 'යටතේ' වෙනුවට 'යටපේ', හෝ අනවශ්‍ය 'ස්' අකුරු එකතු වීම වැනි ය. මෙය "
    "සාමාන්‍ය තත්ත්වයකි — එවැනි දෝෂ තිබූ පමණින් ඡේදය නොසලකා හරින්න එපා. ඔබේ සිංහල භාෂා දැනුම "
    "යොදාගෙන එම අකුරු මට්ටමේ දෝෂ මානසිකව නිවැරදි කර, ඡේදයේ සැබවින් ම අඩංගු කරුණු — විශේෂයෙන් ම "
    "සංඛ්‍යා, මුදල් ප්‍රමාණ, දින, වර්ෂ (මේවා අකුරු දෝෂවලින් බලපෑමට ලක් නොවේ, එබැවින් සැම විටම "
    "විශ්වාස කළ හැක) — හඳුනාගෙන පිළිතුර සකසන්න.\n\n"
    "සන්දර්භයේ විවිධ ඛණ්ඩවල සැබැවින් ම සඳහන් තොරතුරු ඒකාබද්ධ කර සම්පූර්ණ පිළිතුරක් ලෙස ඉදිරිපත් "
    "කළ හැක — සියලු තොරතුරු එක් ඛණ්ඩයක ම තිබිය යුතු නැත. නමුත් සන්දර්භයේ නොමැති කිසිදු නව "
    "කරුණක්, විස්තරයක් හෝ උදාහරණයක් ඔබේ පොදු දැනුමෙන් හෝ අනුමානයෙන් එකතු නොකරන්න — අකුරු මට්ටමේ "
    "දෝෂ නිවැරදි කිරීම පමණක් අවසර ඇත, නව අන්තර්ගතයක් නිර්මාණය කිරීම නොවේ.\n\n"
    f"පහත අවස්ථා දෙකේදී පමණක්, හරියටම මෙසේ පිළිතුරු දෙන්න — වෙනත් කිසිවක් නොලියා: '{REFUSAL}'\n"
    "1. සන්දර්භය කිසිසේත් සිංහල හෝ වෙනත් තේරුම් ගත හැකි භාෂාවක් නොවේ නම් — එනම් එහි එක "
    "වචනයක්වත් හඳුනාගත නොහැකි, සම්පූර්ණයෙන් අහඹු අකුරු/සංකේත මාලාවක් නම් පමණි (ලතින් අකුරු "
    "වලින් හෝ වැරදි කේතනයකින් ලැබුණු, කිසිසේත් තේරුම් ගත නොහැකි පෙළක් මෙයට උදාහරණයකි).\n"
    "2. දෝෂ සහිත වුවත්, සන්දර්භයේ ප්‍රශ්නයට අදාළ කිසිදු තොරතුරක් නොමැති නම්.\n\n"
    "සැක සහිත විටෙක පවා, දෝෂ සහිත පෙළින් වුවත් ප්‍රශ්නයට අදාළ නිශ්චිත කරුණක් (සංඛ්‍යාවක්, "
    "මුදලක්, කාල සීමාවක් වැනි) හඳුනාගත හැකි නම්, එය භාවිත කර පිළිතුරු දීමට උත්සාහ කරන්න — "
    f"ප්‍රතික්ෂේප කිරීම අවසාන විකල්පය පමණි."
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

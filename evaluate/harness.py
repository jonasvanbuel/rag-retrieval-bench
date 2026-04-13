"""Evaluation harness.

Evaluates one (question, retrieved_chunks) pair and returns a results.json entry.

Metrics:
  faithfulness          — RAGAS: are answer claims supported by the context?
  context_precision     — RAGAS: is the retrieved context relevant?
  context_recall        — RAGAS: does the context cover the ground truth?
  answer_relevance      — cosine(embed(question), embed(answer)); no extra LLM call
  uncertainty_appropriate — custom LLM judge for unanswerable questions only

LLM backend: Groq (`llama-3.3-70b-versatile`) via `GROQ_API_KEY`.
`runner.py` validates the key before evaluation; `--no-eval` skips all LLM calls.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embedders import BaseEmbedder
    from retrieval import RetrievedChunk

import numpy as np

# ---------------------------------------------------------------------------
# LLM backend (Groq only)
# ---------------------------------------------------------------------------

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.3-70b-versatile"


def _require_groq() -> None:
    if not (GROQ_API_KEY and GROQ_API_KEY.strip()):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to `.env` (see .env.example) or run with --no-eval."
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_PROMPT = (
    "Answer the following question based ONLY on the provided context. "
    "If the context does not contain sufficient information to answer, "
    "say so explicitly — do not guess.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

UNCERTAINTY_PROMPT = (
    "Question: {question}\n\n"
    "Answer: {answer}\n\n"
    "Does this answer appropriately express uncertainty or inability to answer "
    "(rather than asserting a specific fact)? "
    "Reply with exactly one word: YES or NO."
)


# ---------------------------------------------------------------------------
# LLM backend: RAGAS wrapper
# ---------------------------------------------------------------------------

def _get_ragas_llm():
    """Return a RAGAS-compatible LangchainLLMWrapper (Groq OpenAI-compatible API)."""
    _require_groq()
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    return LangchainLLMWrapper(ChatOpenAI(
        model=GROQ_MODEL, base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY, temperature=0,
    ))


# ---------------------------------------------------------------------------
# LLM backend: answer generation
# ---------------------------------------------------------------------------

def generate_answer(question: str, contexts: list[str],
                    max_retries: int = 3) -> tuple[str, float]:
    """Generate an answer from the retrieved contexts.  Returns (answer, latency_ms).

    Retries up to max_retries times if the LLM returns an empty response.
    """
    context = "\n\n---\n\n".join(contexts)
    prompt  = ANSWER_PROMPT.format(context=context, question=question)

    _require_groq()
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    for attempt in range(1, max_retries + 1):
        t0   = time.perf_counter()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0,
        )
        answer  = resp.choices[0].message.content.strip()
        elapsed = (time.perf_counter() - t0) * 1000

        if answer:
            return answer, elapsed

        print(f"    [warn] Empty answer on attempt {attempt}/{max_retries} — retrying...",
              flush=True)
        time.sleep(1)

    print(f"    [warn] All {max_retries} attempts returned empty answer; recording blank.",
          flush=True)
    return "", elapsed


# ---------------------------------------------------------------------------
# LLM backend: uncertainty judge
# ---------------------------------------------------------------------------

def _judge_uncertainty(question: str, answer: str) -> bool:
    """LLM judge: does the answer express appropriate uncertainty?"""
    _require_groq()
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content":
            UNCERTAINTY_PROMPT.format(question=question, answer=answer)}],
        max_tokens=5, temperature=0,
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# RAGAS metric evaluation
# ---------------------------------------------------------------------------

async def _run_ragas_metrics(sample, llm_wrapper) -> dict[str, float | None]:
    """Run Faithfulness, ContextPrecision, ContextRecall on one sample."""
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    metrics = [
        Faithfulness(llm=llm_wrapper),
        LLMContextPrecisionWithReference(llm=llm_wrapper),
        LLMContextRecall(llm=llm_wrapper),
    ]
    scores: dict[str, float | None] = {}
    for metric in metrics:
        try:
            score = await metric.single_turn_ascore(sample)
            scores[metric.name] = float(score) if score is not None else None
        except Exception as exc:
            print(f"    [warn] RAGAS {metric.name} failed: {exc}")
            scores[metric.name] = None
    return scores


def _ragas_metric_names() -> dict[str, str]:
    """Map RAGAS internal names → our results schema field names."""
    return {
        "faithfulness":                         "faithfulness",
        "llm_context_precision_with_reference": "context_precision",
        "context_recall":                       "context_recall",
    }


def run_ragas(question: str, answer: str, contexts: list[str],
              reference: str) -> dict[str, float | None]:
    """Synchronous wrapper around the async RAGAS evaluation."""
    try:
        from ragas import SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import SingleTurnSample  # type: ignore[no-redef]

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )
    llm_wrapper = _get_ragas_llm()
    raw_scores  = asyncio.run(_run_ragas_metrics(sample, llm_wrapper))

    name_map = _ragas_metric_names()
    return {our_name: raw_scores.get(ragas_name)
            for ragas_name, our_name in name_map.items()}


# ---------------------------------------------------------------------------
# Answer relevance (cosine similarity — no extra LLM call)
# ---------------------------------------------------------------------------

def compute_answer_relevance(question: str, answer: str,
                             embedder: "BaseEmbedder") -> float:
    """Cosine similarity between question and answer embeddings."""
    if not answer or not answer.strip():
        return 0.0
    q_vec = embedder.embed_query(question)
    a_vec = embedder.embed_query(answer)
    return float(np.dot(q_vec, a_vec))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_question(
    question: dict,
    retrieved: list["RetrievedChunk"],
    embedder: "BaseEmbedder",
    retrieval_ms: float,
    config: dict,
    run_llm_eval: bool = True,
) -> dict:
    """Evaluate one question under one config.  Returns a results.json entry."""
    is_unanswerable = question["type"] == "unanswerable"
    source_papers   = question.get("source_papers", [])
    retrieved_papers = [rc.chunk.paper_id for rc in retrieved]
    contexts         = [rc.chunk.text for rc in retrieved]
    chunk_ids        = [f"{rc.chunk.paper_id}:{rc.chunk.chunk_index}" for rc in retrieved]

    if run_llm_eval:
        answer, gen_ms = generate_answer(question["question"], contexts)
    else:
        answer, gen_ms = "[evaluation skipped — run without --no-eval]", 0.0

    metrics: dict[str, float | bool | None] = {
        "context_precision":       None,
        "context_recall":          None,
        "faithfulness":            None,
        "answer_relevance":        None,
        "uncertainty_appropriate": None,
    }

    if run_llm_eval:
        if not is_unanswerable:
            ragas_scores = run_ragas(
                question=question["question"],
                answer=answer,
                contexts=contexts,
                reference=question["reference_answer"],
            )
            metrics.update(ragas_scores)
            metrics["answer_relevance"] = compute_answer_relevance(
                question["question"], answer, embedder
            )
        else:
            metrics["uncertainty_appropriate"] = _judge_uncertainty(
                question["question"], answer
            )

    return {
        "config":               config,
        "question_id":          question["id"],
        "question_type":        question["type"],
        "question":             question["question"],
        "retrieved_chunk_ids":  chunk_ids,
        "retrieved_paper_ids":  retrieved_papers,
        "source_paper_ids":     source_papers,
        "generated_answer":     answer,
        "metrics":              metrics,
        "latency": {
            "retrieval_ms":   round(retrieval_ms, 1),
            "generation_ms":  round(gen_ms, 1),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

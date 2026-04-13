#!/usr/bin/env python3
"""RAG retrieval benchmark runner.

Evaluates all (or a filtered subset of) 36 configurations against the question set.
Results are written incrementally to results/results.json; interrupted runs can be
resumed — already-evaluated (config, question) pairs are skipped automatically.
When the run finishes, the HTML report is generated (results/report.html) and opened
in the default browser.

Usage:
  python runner.py                                               # all configs (needs GROQ + OPENAI keys)
  python runner.py --chunker fixed_size --embedder minilm --retriever semantic
  python runner.py --chunker fixed_size                          # 12 configs
  python runner.py --no-eval                                     # retrieval only — Groq key not required
  python runner.py --top-k 10

Required API keys depend on flags; missing keys cause an immediate error. See README.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_FILE   = Path("results/results.json")
QUESTIONS_FILE = Path("questions/question_set.json")
PAPERS_DIR     = Path("data")
# Default retrieval depth (chunks per query). Override per run: python runner.py --top-k 10
TOP_K_DEFAULT  = 5

# All 36 configurations: 3 chunkers × 3 embedders × 4 retrieval strategies
ALL_CONFIGS = [
    {"chunker": c, "embedder": e, "retriever": r}
    for c in ("fixed_size", "sentence", "semantic")
    for e in ("minilm", "bge_m3", "openai")
    for r in ("semantic", "bm25", "hybrid", "reranker")
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_papers() -> dict[str, str]:
    papers = {}
    for md_file in sorted(PAPERS_DIR.glob("*.md")):
        papers[md_file.stem] = md_file.read_text(encoding="utf-8")
    return papers


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_FILE.read_text())["questions"]


def load_existing_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    try:
        return json.loads(RESULTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_result(entry: dict, all_results: list[dict]) -> None:
    all_results.append(entry)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_chunker(name: str):
    from chunkers.fixed_size import FixedSizeChunker
    from chunkers.sentence   import SentenceChunker
    from chunkers.semantic   import SemanticChunker
    return {"fixed_size": FixedSizeChunker,
            "sentence":   SentenceChunker,
            "semantic":   SemanticChunker}[name]()


def get_embedder(name: str):
    from embedders.minilm       import MiniLMEmbedder
    from embedders.bge_m3       import BGEM3Embedder
    from embedders.openai_embed import OpenAIEmbedder
    return {"minilm":  MiniLMEmbedder,
            "bge_m3":  BGEM3Embedder,
            "openai":  OpenAIEmbedder}[name]()


def build_retriever(config: dict, chunks, embeddings, embedder, top_k: int):
    from retrieval.semantic  import SemanticRetriever
    from retrieval.bm25      import BM25Retriever
    from retrieval.hybrid    import HybridRetriever
    from retrieval.reranker  import RerankerRetriever

    sem = SemanticRetriever(chunks, embeddings, embedder)
    bm  = BM25Retriever(chunks)
    name = config["retriever"]
    if name == "semantic":  return sem
    if name == "bm25":      return bm
    hyb = HybridRetriever(sem, bm)
    if name == "hybrid":    return hyb
    return RerankerRetriever(hyb)


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def _env_nonempty(name: str) -> bool:
    v = os.getenv(name)
    return bool(v and v.strip())


def validate_api_keys(configs_to_run: list[dict], no_eval: bool) -> None:
    """Exit unless this run has the API keys it needs.

    Full grid (36 configs, with evaluation) requires both GROQ_API_KEY and
    OPENAI_API_KEY. Narrower runs need only the keys for the embedders and
    whether evaluation runs (see --no-eval).
    """
    missing: list[tuple[str, str]] = []
    if not no_eval and not _env_nonempty("GROQ_API_KEY"):
        missing.append((
            "GROQ_API_KEY",
            "Required for answer generation and RAGAS evaluation "
            "(free tier: https://console.groq.com). "
            "Use --no-eval to skip evaluation and omit this key.",
        ))
    if any(c["embedder"] == "openai" for c in configs_to_run) and not _env_nonempty("OPENAI_API_KEY"):
        missing.append((
            "OPENAI_API_KEY",
            "Required for the OpenAI embedding model (text-embedding-3-small) "
            "in this run. Use e.g. --embedder minilm to run without it.",
        ))
    if not missing:
        return
    print("\n[error] Missing required API key(s). Set them in `.env` (see .env.example):\n")
    for name, detail in missing:
        print(f"  • {name}\n    {detail}\n")
    print("Questions about this benchmark: jonas@vanbuel.dev\n")
    sys.exit(1)


def announce_llm_backend(no_eval: bool) -> None:
    if no_eval:
        return
    print("[llm] Using Groq llama-3.3-70b-versatile for evaluation")
    print("      (~2h for a full run). Free tier at https://console.groq.com")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG retrieval benchmark — evaluate chunker × embedder × retriever configs"
    )
    parser.add_argument("--chunker",   choices=["fixed_size", "sentence", "semantic"],
                        help="Run only configs with this chunker")
    parser.add_argument("--embedder",  choices=["minilm", "bge_m3", "openai"],
                        help="Run only configs with this embedder")
    parser.add_argument("--retriever", choices=["semantic", "bm25", "hybrid", "reranker"],
                        help="Run only configs with this retriever")
    parser.add_argument("--top-k",    type=int, default=TOP_K_DEFAULT,
                        help=f"Number of chunks to retrieve (default: {TOP_K_DEFAULT})")
    parser.add_argument("--no-eval",  action="store_true",
                        help="Skip LLM evaluation — collect retrieval metadata only (faster)")
    args = parser.parse_args()

    # Filter configs
    configs_to_run = [
        c for c in ALL_CONFIGS
        if (args.chunker   is None or c["chunker"]   == args.chunker)
        and (args.embedder  is None or c["embedder"]  == args.embedder)
        and (args.retriever is None or c["retriever"] == args.retriever)
    ]
    if not configs_to_run:
        print("[error] No configs match the given filters.")
        sys.exit(1)

    validate_api_keys(configs_to_run, args.no_eval)
    announce_llm_backend(args.no_eval)

    questions = load_questions()
    papers    = load_papers()
    print(f"\nLoaded {len(papers)} papers · {len(questions)} questions · "
          f"{len(configs_to_run)} config(s) · top_k={args.top_k}\n")

    # ------------------------------------------------------------------
    # Phase 1: Pre-build chunk cache and embedding cache
    # ------------------------------------------------------------------
    import numpy as np

    needed_chunkers    = {c["chunker"]  for c in configs_to_run}
    needed_embed_pairs = {(c["chunker"], c["embedder"]) for c in configs_to_run}

    chunk_cache:   dict[str, list]         = {}
    embed_cache:   dict[tuple, np.ndarray] = {}
    embedder_cache: dict[str, object]      = {}

    # DEV CACHE — speeds up repeated runs during development.
    # Delete .cache/ to force recompute. Strip this block before release.
    import pickle
    CACHE_DIR = Path(".cache")
    CACHE_DIR.mkdir(exist_ok=True)

    for chunker_name in sorted(needed_chunkers):
        cache_file = CACHE_DIR / f"chunks_{chunker_name}.pkl"
        if cache_file.exists():
            print(f"Chunking   [{chunker_name}]... (cached)", end=" ", flush=True)
            with cache_file.open("rb") as f:
                chunk_cache[chunker_name] = pickle.load(f)
            print(f"→ {len(chunk_cache[chunker_name]):,} chunks")
        else:
            chunker = get_chunker(chunker_name)
            print(f"Chunking   [{chunker_name}]...", end=" ", flush=True)
            t0 = time.perf_counter()
            chunk_cache[chunker_name] = chunker.chunk_all(papers)
            elapsed = time.perf_counter() - t0
            with cache_file.open("wb") as f:
                pickle.dump(chunk_cache[chunker_name], f)
            print(f"→ {len(chunk_cache[chunker_name]):,} chunks  ({elapsed:.1f}s)")

    print()
    for chunker_name, embedder_name in sorted(needed_embed_pairs):
        cache_file = CACHE_DIR / f"embeddings_{chunker_name}_{embedder_name}.npy"
        if cache_file.exists():
            print(f"Embedding  [{chunker_name} + {embedder_name}]... (cached)")
            embed_cache[(chunker_name, embedder_name)] = np.load(str(cache_file))
        else:
            if embedder_name not in embedder_cache:
                embedder_cache[embedder_name] = get_embedder(embedder_name)
            embedder = embedder_cache[embedder_name]
            chunks   = chunk_cache[chunker_name]
            texts    = [c.text for c in chunks]
            print(f"Embedding  [{chunker_name} + {embedder_name}]  ({len(texts):,} chunks)...",
                  end=" ", flush=True)
            t0 = time.perf_counter()
            matrix = embedder.embed(texts)
            np.save(str(cache_file), matrix)
            embed_cache[(chunker_name, embedder_name)] = matrix
            print(f"done ({time.perf_counter()-t0:.1f}s)")

    # Ensure all embedder instances are loaded (cached embed runs skip get_embedder)
    for chunker_name, embedder_name in sorted(needed_embed_pairs):
        if embedder_name not in embedder_cache:
            embedder_cache[embedder_name] = get_embedder(embedder_name)

    # ------------------------------------------------------------------
    # Phase 2: Evaluation loop
    # ------------------------------------------------------------------
    from evaluate.harness import evaluate_question

    all_results = load_existing_results()
    completed   = {
        (r["config"]["chunker"], r["config"]["embedder"],
         r["config"]["retriever"], r["question_id"])
        for r in all_results
    }

    total_configs = len(configs_to_run)
    for cfg_num, config in enumerate(configs_to_run, start=1):
        label = f"{config['chunker']} + {config['embedder']} + {config['retriever']}"
        print(f"\nConfig {cfg_num:>2}/{total_configs}: {label}")

        chunks     = chunk_cache[config["chunker"]]
        embeddings = embed_cache[(config["chunker"], config["embedder"])]
        embedder   = embedder_cache[config["embedder"]]
        retriever  = build_retriever(config, chunks, embeddings, embedder, args.top_k)

        latencies: list[float] = []
        for question in questions:
            key = (config["chunker"], config["embedder"],
                   config["retriever"], question["id"])
            if key in completed:
                continue

            t0        = time.perf_counter()
            retrieved = retriever.retrieve(question["question"], top_k=args.top_k)
            ret_ms    = (time.perf_counter() - t0) * 1000

            print(f"  {question['id']} [{question['type']:<12}]", end=" ", flush=True)

            entry = evaluate_question(
                question=question,
                retrieved=retrieved,
                embedder=embedder,
                retrieval_ms=ret_ms,
                config=config,
                run_llm_eval=not args.no_eval,
            )
            gen_ms = entry["latency"]["generation_ms"]
            latencies.append(ret_ms + gen_ms)
            print(f"ret {ret_ms:>5.0f}ms  gen {gen_ms:>6.0f}ms")

            save_result(entry, all_results)
            completed.add(key)

        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"  ✓ config complete — avg {avg:.0f}ms/question")
        else:
            print("  ✓ all questions already evaluated (skipped)")

    print(f"\nDone. {len(all_results)} total results in {RESULTS_FILE}")

    report_script = Path(__file__).resolve().parent / "scripts" / "generate_report.py"
    print("\nGenerating HTML report...")
    subprocess.run([sys.executable, str(report_script)], check=True)


if __name__ == "__main__":
    main()

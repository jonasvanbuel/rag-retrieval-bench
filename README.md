# RAG Retrieval Bench

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

There is no universally best RAG configuration—what works on one dataset can underperform on another—so trade-offs only become clear when you measure. This repository runs a controlled **3 × 3 × 4 = 36** grid: chunking, embedding, and retrieval vary while the corpus and questions stay fixed, so differences reflect pipeline choices. The point is to make those trade-offs visible, not to pick one winner for every use case.

📖 You can find a more detailed write-up here: [https://vanbuel.dev/projects/rag-retrieval-bench/](https://vanbuel.dev/projects/rag-retrieval-bench/)

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## API keys

Two API keys are required. Copy the example file and set both:

```bash
cp .env.example .env
```

- `OPENAI_API_KEY` — used by the `text-embedding-3-small` embedder
- `GROQ_API_KEY` — used by `llama-3.3-70b-versatile` for answer generation and RAGAS evaluation

## Full run

A **full run** evaluates all **36** configurations (all chunkers, embedders, and retrievers) with LLM evaluation.

On macOS, `caffeinate` avoids sleep during long jobs.

```bash
caffeinate -id python runner.py
```

Results stream to the terminal and append to `results/results.json`. When the run finishes, the runner generates `results/report.html` and opens it in your default browser.

### Useful flags

Filters apply together: only configurations matching **all** specified dimensions are run:

- Chunker values: `fixed_size`, `sentence`, `semantic`
- Embedder values: `minilm`, `bge_m3`, `openai`
- Retriever values: `semantic`, `bm25`, `hybrid`, `reranker`

```bash
# Single configuration
python runner.py --chunker fixed_size --embedder minilm --retriever semantic

# All configs that use one chunker (12 runs: 3 embedders × 4 retrievers)
python runner.py --chunker sentence

# Local embedders only — no OPENAI_API_KEY
python runner.py --embedder bge_m3

# Retrieval and timing only — no Groq / RAGAS (no GROQ_API_KEY)
python runner.py --no-eval

# Retrieve more chunks per query (default is 5)
python runner.py --top-k 10 --chunker fixed_size --embedder minilm --retriever hybrid
```

## Benchmarking your own data

### 1. Data

Place one markdown file per document in [`data/`](data/), named `*.md`. The stem of each filename is used as the document id. Replace or add files here to benchmark your own material instead of the sample papers.

### 2. Question set

Edit [`questions/question_set.json`](questions/question_set.json). Keep the same JSON shape as the sample file. Align wording and references with how real users would query *your* documents.

After updating both, run `python runner.py` (optionally with the flags above) so retrieval and scores reflect your data and task.

## License

MIT

#!/usr/bin/env python3
"""Generate an HTML report from results/results.json.

Reads the benchmark results, aggregates metrics per configuration,
renders a self-contained HTML report via Jinja2, and opens it in the browser.

Usage (from repository root):
  python scripts/generate_report.py
  python scripts/generate_report.py --results path/to/results.json
  python scripts/generate_report.py --no-open   # write report but don't auto-open
"""
import argparse
import json
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "results/results.json"
REPORT_FILE = REPO_ROOT / "results/report.html"
TEMPLATE_DIR = REPO_ROOT / "templates"
TEMPLATE_NAME = "report.html.j2"

METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevance"]

CHUNKER_COLORS: dict[str, str] = {
    "fixed_size": "#2563eb",
    "sentence":   "#d97706",
    "semantic":   "#dc2626",
}

# Scatter plot canvas constants (build_scatter_block)
_SC_ML = 68   # left margin (space for Y-axis labels)
_SC_MR = 28   # right margin


def aggregate(results: list[dict]) -> list[dict]:
    """Group by config tuple, compute mean for each metric."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        cfg = r["config"]
        key = (cfg["chunker"], cfg["embedder"], cfg["retriever"])
        groups[key].append(r)

    rows = []
    for (chunker, embedder, retriever), entries in sorted(groups.items()):
        def mean_metric(name: str) -> float | None:
            vals = [e["metrics"][name] for e in entries
                    if e["metrics"].get(name) is not None]
            return round(float(np.mean(vals)), 3) if vals else None

        def mean_latency(key: str) -> float:
            vals = [e["latency"][key] for e in entries]
            return round(float(np.mean(vals)), 1) if vals else 0.0

        uncertainty_entries = [e for e in entries
                               if e["question_type"] == "unanswerable"
                               and e["metrics"].get("uncertainty_appropriate") is not None]
        uncertainty_rate = (
            round(sum(1 for e in uncertainty_entries
                      if e["metrics"]["uncertainty_appropriate"]) / len(uncertainty_entries), 3)
            if uncertainty_entries else None
        )

        # Overall average across the 4 main metrics
        metric_vals = [mean_metric(m) for m in METRICS]
        overall = round(float(np.mean([v for v in metric_vals if v is not None])), 3) if any(v is not None for v in metric_vals) else None

        rows.append({
            "chunker":              chunker,
            "embedder":             embedder,
            "retriever":            retriever,
            "n":                    len(entries),
            "context_precision":    mean_metric("context_precision"),
            "context_recall":       mean_metric("context_recall"),
            "faithfulness":         mean_metric("faithfulness"),
            "answer_relevance":     mean_metric("answer_relevance"),
            "uncertainty_rate":     uncertainty_rate,
            "avg_retrieval_ms":     mean_latency("retrieval_ms"),
            "avg_generation_ms":    mean_latency("generation_ms"),
            "overall":              overall,
        })
    return rows


def find_bests(rows: list[dict]) -> dict[str, float]:
    """Find the best (max) value per metric across all rows."""
    all_metrics = METRICS + ["uncertainty_rate", "overall"]
    bests: dict[str, float] = {}
    for m in all_metrics:
        vals = [r[m] for r in rows if r[m] is not None]
        if vals:
            bests[m] = max(vals)
    return bests


def component_averages(summary: list[dict], component: str) -> dict[str, dict]:
    """Average metrics across all configs sharing the same component value."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in summary:
        groups[row[component]].append(row)

    result = {}
    for name, rows in groups.items():
        vals = {m: [r[m] for r in rows if r[m] is not None] for m in METRICS}
        avgs = {m: round(float(np.mean(v)), 3) for m, v in vals.items() if v}
        avgs["overall"] = round(float(np.mean(list(avgs.values()))), 3) if avgs else 0.0
        # Add latency for retriever comparison
        latency_vals = [r["avg_retrieval_ms"] for r in rows]
        avgs["latency"] = round(float(np.mean(latency_vals)), 0) if latency_vals else 0
        result[name] = avgs
    return result


def build_scatter_block(
    summary: list[dict],
    display_names: dict[str, str],
    chunker_colors: dict[str, str] = CHUNKER_COLORS,
) -> str:
    """Return a self-contained HTML block: SVG scatter + tooltip div + script.

    The legend and caption live inside the SVG so they always centre relative
    to the plot area — independent of the width of any surrounding container.
    The containing element must have position:relative for the tooltip.
    """
    W          = 760
    ML, MR     = _SC_ML, _SC_MR
    MT         = 16
    PLOT_BTM   = 360             # bottom of plot area (fixed, not derived from H)
    pw         = W - ML - MR    # 664 — plot width
    ph         = PLOT_BTM - MT  # 344 — plot height
    PCX        = W // 2         # 380 — visual centre of the SVG rectangle
    x_min, x_max = 0, 720
    y_min, y_max = 0.49, 0.85

    # Space below plot area:
    #   16px  x-tick labels     → PLOT_BTM + 16
    #   36px  x-axis label      → PLOT_BTM + 52
    #   46px  legend row        → PLOT_BTM + 98
    #   34px  caption           → PLOT_BTM + 132
    #   18px  bottom pad
    # Total below: 150px  →  H = PLOT_BTM + 150 = 510
    H = PLOT_BTM + 150

    XAXIS_LABEL_Y = PLOT_BTM + 52
    LEGEND_Y      = PLOT_BTM + 98
    CAPTION_Y     = PLOT_BTM + 132

    def pxc(x: float) -> float:
        return ML + (x - x_min) / (x_max - x_min) * pw

    def pyc(y: float) -> float:
        return PLOT_BTM - (y - y_min) / (y_max - y_min) * ph

    els: list[str] = []

    # Background
    els.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')

    # Y grid + labels
    for i in range(8):
        yt = round(0.50 + i * 0.05, 2)
        ty = pyc(yt)
        els.append(f'<line x1="{ML}" y1="{ty:.1f}" x2="{W - MR}" y2="{ty:.1f}" stroke="#f0f1f5" stroke-width="1"/>')
        els.append(f'<text x="{ML - 8}" y="{ty:.1f}" text-anchor="end" dominant-baseline="middle" font-size="11" fill="#aaa">{yt:.2f}</text>')

    # X grid + labels
    for xt in [0, 100, 200, 300, 400, 500, 600, 700]:
        tx = pxc(xt)
        els.append(f'<line x1="{tx:.1f}" y1="{MT}" x2="{tx:.1f}" y2="{PLOT_BTM}" stroke="#f0f1f5" stroke-width="1"/>')
        els.append(f'<text x="{tx:.1f}" y="{PLOT_BTM + 16}" text-anchor="middle" font-size="11" fill="#aaa">{xt}</text>')

    # Axes
    els.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{PLOT_BTM}" stroke="#d0d3da" stroke-width="1.5"/>')
    els.append(f'<line x1="{ML}" y1="{PLOT_BTM}" x2="{W - MR}" y2="{PLOT_BTM}" stroke="#d0d3da" stroke-width="1.5"/>')

    # X-axis label — centred at SVG midpoint
    els.append(
        f'<text x="{PCX}" y="{XAXIS_LABEL_Y}" text-anchor="middle"'
        f' font-size="12" fill="#555" font-weight="500">'
        f'Per-query retrieval latency (ms)*</text>'
    )
    # Y-axis label
    cy_label = MT + ph // 2
    els.append(
        f'<text x="14" y="{cy_label}" text-anchor="middle"'
        f' font-size="12" fill="#555" font-weight="500"'
        f' transform="rotate(-90, 14, {cy_label})">Overall score</text>'
    )

    # Dots + rank badges
    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "\u2014"

    for rank, row in enumerate(summary, 1):
        if row["overall"] is None:
            continue
        dot_cx = pxc(row["avg_retrieval_ms"])
        dot_cy = pyc(row["overall"])
        color   = chunker_colors.get(row["chunker"], "#999")
        label   = (
            f'{display_names.get(row["chunker"],  row["chunker"])} \u00b7 '
            f'{display_names.get(row["embedder"], row["embedder"])} \u00b7 '
            f'{display_names.get(row["retriever"],row["retriever"])}'
        )
        data = (
            f' data-config="{label}"'
            f' data-score="{row["overall"]:.3f}"'
            f' data-latency="{int(row["avg_retrieval_ms"])}"'
            f' data-cp="{_fmt(row["context_precision"])}"'
            f' data-cr="{_fmt(row["context_recall"])}"'
            f' data-fa="{_fmt(row["faithfulness"])}"'
            f' data-ar="{_fmt(row["answer_relevance"])}"'
        )
        els.append(
            f'<circle cx="{dot_cx:.1f}" cy="{dot_cy:.1f}" r="8"'
            f' fill="{color}" fill-opacity="0.75" stroke="{color}" stroke-width="1.5"'
            f' style="cursor:pointer"{data}/>'
        )
        if rank <= 3:
            bx, by = dot_cx + 13, dot_cy - 13
            els.append(
                f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="9" fill="#1a6b2a"'
                f' style="cursor:pointer"{data}/>'
            )
            els.append(
                f'<text x="{bx:.1f}" y="{by:.1f}" text-anchor="middle"'
                f' dominant-baseline="middle" font-size="9" fill="#fff"'
                f' font-weight="700" pointer-events="none">{rank}</text>'
            )

    # Legend — inside SVG, horizontally centred at PCX
    legend_items = [
        (chunker_colors["fixed_size"], "Fixed-size chunker"),
        (chunker_colors["sentence"],   "Sentence chunker"),
        (chunker_colors["semantic"],   "Semantic chunker"),
    ]
    ITEM_R    = 5
    TEXT_GAP  = 7    # gap between circle edge and label start
    ITEM_GAP  = 22   # gap between consecutive items
    CHAR_W    = 6.4  # approximate px-per-char at 11px system font

    item_widths = [ITEM_R * 2 + TEXT_GAP + len(lbl) * CHAR_W for _, lbl in legend_items]
    total_legend_w = sum(item_widths) + ITEM_GAP * (len(legend_items) - 1)
    legend_y  = LEGEND_Y
    cur_x     = PCX - total_legend_w / 2

    for (color, lbl), iw in zip(legend_items, item_widths):
        circ_cx = cur_x + ITEM_R
        text_x  = cur_x + ITEM_R * 2 + TEXT_GAP
        els.append(f'<circle cx="{circ_cx:.1f}" cy="{legend_y}" r="{ITEM_R}" fill="{color}" fill-opacity="0.75"/>')
        els.append(f'<text x="{text_x:.1f}" y="{legend_y}" dominant-baseline="middle" font-size="11" fill="#555">{lbl}</text>')
        cur_x += iw + ITEM_GAP

    # Caption — inside SVG, centred at SVG midpoint
    els.append(
        f'<text x="{PCX}" y="{CAPTION_Y}" text-anchor="middle"'
        f' font-size="11" fill="#aaa">'
        f'* Per-query retrieval latency only \u2014 '
        f'excludes index-build time (chunking and embedding).</text>'
    )

    svg = (
        f'<svg id="rr-scatter" width="{W}" height="{H}" viewBox="0 0 {W} {H}"'
        f' style="display:block;max-width:100%;margin:0 auto;font-family:inherit;">\n'
        + "\n".join(f"  {e}" for e in els)
        + "\n</svg>"
    )

    tip_div = (
        '<div id="rr-tip" style="display:none;position:absolute;background:#fff;'
        'border:1px solid #e2e4e8;border-radius:8px;padding:12px 14px;font-size:12px;'
        'line-height:1.5;box-shadow:0 4px 16px rgba(0,0,0,0.1);pointer-events:none;'
        'min-width:210px;z-index:20;"></div>'
    )

    script = """\
<script>
(function () {
  var svg = document.getElementById('rr-scatter');
  var tip = document.getElementById('rr-tip');
  if (!svg || !tip) return;
  var wrap = tip.parentElement;
  function tr(label, val, color, bold) {
    return '<tr><td style="color:#888;padding:2px 16px 2px 0;border:none">' + label + '</td>' +
      '<td style="text-align:right;font-family:monospace;border:none' +
      (bold ? ';font-weight:700' : '') + (color ? ';color:' + color : '') +
      '">' + val + '</td></tr>';
  }
  function sep() {
    return '<tr><td colspan="2" style="padding:4px 0 3px">' +
      '<hr style="border:none;border-top:1px solid #f0f1f5;margin:0"/></td></tr>';
  }
  svg.querySelectorAll('circle[data-score]').forEach(function (dot) {
    dot.addEventListener('mouseenter', function () {
      tip.innerHTML =
        '<div style="font-weight:600;color:#111;margin-bottom:8px;line-height:1.4">' +
        dot.dataset.config + '</div>' +
        '<table style="width:100%;border-collapse:collapse;font-size:11px">' +
        tr('Overall score',      dot.dataset.score,   '#1a6b2a', true) +
        tr('Context Precision',  dot.dataset.cp) +
        tr('Context Recall',     dot.dataset.cr) +
        tr('Faithfulness',       dot.dataset.fa) +
        tr('Answer Relevance',   dot.dataset.ar) +
        tr('Retrieval latency',  dot.dataset.latency + '\u202fms') +
        '</table>';
      tip.style.display = 'block';
    });
    dot.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
    dot.addEventListener('mousemove', function (e) {
      var rect = wrap.getBoundingClientRect();
      var x = e.clientX - rect.left + 18;
      var y = e.clientY - rect.top - 10;
      if (x + 240 > wrap.offsetWidth) x = e.clientX - rect.left - 256;
      if (y < 0) y = 4;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    });
  });
}());
</script>"""

    return "\n".join([svg, tip_div, script])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS_FILE))
    parser.add_argument("--no-open", action="store_true",
                        help="Write report but don't open it in the browser")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(
            f"[error] {results_path} not found. Run `python runner.py` first, "
            f"or pass --results to an existing results.json."
        )
        raise SystemExit(1)

    results = json.loads(results_path.read_text())
    if not results:
        print("[error] results.json is empty.")
        raise SystemExit(1)

    print(f"Loaded {len(results)} result entries")

    # Overall summary — sorted best-first
    summary = sorted(aggregate(results), key=lambda r: r["overall"] or 0, reverse=True)
    bests   = find_bests(summary)

    # Per question-type breakdowns — each sorted best-first (no unanswerable tab in report)
    qtypes = ["factual", "conceptual", "multi_hop"]
    breakdowns: dict[str, list[dict]] = {}
    breakdown_bests: dict[str, dict] = {}
    for qtype in qtypes:
        filtered = [r for r in results if r["question_type"] == qtype]
        rows = sorted(aggregate(filtered), key=lambda r: r["overall"] or 0, reverse=True)
        breakdowns[qtype] = rows
        breakdown_bests[qtype] = find_bests(rows)

    # Component averages
    chunker_avgs = component_averages(summary, "chunker")
    embedder_avgs = component_averages(summary, "embedder")
    retriever_avgs = component_averages(summary, "retriever")

    # Human-readable display names for raw config identifiers
    display_names: dict[str, str] = {
        "fixed_size":      "Fixed-size",
        "sentence":        "Sentence",
        "semantic":        "Semantic",
        "minilm":          "MiniLM",
        "bge_m3":          "BGE-M3",
        "openai":          "OpenAI",
        "bm25":            "BM25",
        "hybrid":          "Hybrid (RRF)",
        "reranker":        "Reranker",
        "semantic_search": "Semantic",
    }

    # Best component names
    best_chunker = max(chunker_avgs, key=lambda k: chunker_avgs[k]["overall"])
    best_embedder = max(embedder_avgs, key=lambda k: embedder_avgs[k]["overall"])
    best_retriever = max(retriever_avgs, key=lambda k: retriever_avgs[k]["overall"])

    # Max score for bar chart scaling
    all_overalls = [r["overall"] for r in summary if r["overall"] is not None]
    max_score = max(all_overalls) if all_overalls else 1.0

    scatter_block = build_scatter_block(summary, display_names)

    env      = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                           autoescape=True)
    template = env.get_template(TEMPLATE_NAME)
    html     = template.render(
        summary=summary,
        bests=bests,
        breakdowns=breakdowns,
        breakdown_bests=breakdown_bests,
        qtypes=qtypes,
        generated_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        display_names=display_names,
        scatter_block=scatter_block,
        component_avgs=chunker_avgs,
        component_avgs_embedder=embedder_avgs,
        component_avgs_retriever=retriever_avgs,
        component_order={
            "chunkers": sorted(chunker_avgs.keys(), key=lambda k: chunker_avgs[k]["overall"], reverse=True),
            "embedders": sorted(embedder_avgs.keys(), key=lambda k: embedder_avgs[k]["overall"], reverse=True),
            "retrievers": sorted(retriever_avgs.keys(), key=lambda k: retriever_avgs[k]["overall"], reverse=True),
        },
        best_chunker=best_chunker,
        best_embedder=best_embedder,
        best_retriever=best_retriever,
        max_score=max_score,
    )

    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(html, encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")

    if not args.no_open:
        webbrowser.open(REPORT_FILE.resolve().as_uri())


if __name__ == "__main__":
    main()

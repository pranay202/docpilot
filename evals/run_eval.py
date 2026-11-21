"""Offline evaluation: retrieval hit rate, answer accuracy and citation precision.

    # LibreOffice help (real brand, 4 products, pinned revision)
    ./scripts/fetch_libreoffice_help.sh
    python -m evals.run_eval --dataset evals/libreoffice.jsonl --docs brand_docs/libreoffice

    # Bundled fictional sample docs (fast unit-level check)
    python -m evals.run_eval

    # Fail (exit 1) below thresholds, as CI does
    python -m evals.run_eval --dataset evals/libreoffice.jsonl --docs brand_docs/libreoffice --min-hit 0.85 --min-accuracy 0.7

Dataset rows: {question, product, expected_section (str or list), expected_terms[], alternatives[[...]]?, expected_source?}.
Metrics per question:
- hit@k:              a top-k chunk comes from the expected section (and source file, when given)
- answer accuracy:    every expected term appears in the answer (case-insensitive)
- citation precision: share of cited chunks that come from the expected section
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.config import load_settings
from app.embeddings import HashingEmbedder
from app.schemas import AskRequest, IngestRequest
from app.service import DocService, _build_llm
from app.store import InMemoryStore

ROOT = Path(__file__).resolve().parent.parent


def load_rows(dataset: Path) -> list[dict]:
    return [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]


def build_service(docs: Path, products: list[str]) -> DocService:
    settings = load_settings()
    svc = DocService(settings, InMemoryStore(), HashingEmbedder(settings.embedding_dim), _build_llm(settings))
    for product in products:
        result = svc.ingest(IngestRequest(product=product, source=str(docs / product)))
        if result.status != "done":
            raise SystemExit(f"Ingest failed for {product}: {result.detail}")
    return svc


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _matches(row: dict, section: str, source: str) -> bool:
    if not any(section.endswith(s) for s in _as_list(row["expected_section"])):
        return False
    return "expected_source" not in row or source.endswith(row["expected_source"])


def _answer_ok(row: dict, answer: str) -> bool:
    """All expected terms present, or every term of any one accepted alternative."""
    a = answer.lower()
    options = [row["expected_terms"], *row.get("alternatives", [])]
    return any(all(t.lower() in a for t in terms) for terms in options)


def evaluate(svc: DocService, rows: list[dict], k: int = 5, use_product: bool = True) -> dict:
    hits = correct = 0
    precisions: list[float] = []
    failures = []
    started = time.perf_counter()
    for row in rows:
        product = row["product"] if use_product else None
        retrieved = svc.search(row["question"], k=k, product=product)
        hit = any(_matches(row, h.chunk.section, h.chunk.source) for h in retrieved)
        resp = svc.ask(AskRequest(question=row["question"], product=product, top_k=k))
        ok = _answer_ok(row, resp.answer)
        cited = resp.citations
        precisions.append(sum(_matches(row, c.section, c.source) for c in cited) / len(cited) if cited else 0.0)
        hits += hit
        correct += ok
        if not (hit and ok):
            failures.append({"question": row["question"], "hit": hit, "answer_ok": ok, "answer": resp.answer[:160]})
    n = len(rows)
    return {
        "questions": n,
        "chunks_indexed": svc.store.count(),
        "mode": "llm" if svc.llm else "extractive",
        "product_hint": use_product,
        f"hit@{k}": round(hits / n, 3),
        "answer_accuracy": round(correct / n, 3),
        "citation_precision": round(sum(precisions) / n, 3),
        "avg_latency_ms": round((time.perf_counter() - started) * 1000 / n, 1),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals" / "dataset.jsonl")
    parser.add_argument("--docs", type=Path, default=ROOT / "sample_docs")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-product", action="store_true", help="do not pass the product hint")
    parser.add_argument("--min-hit", type=float, default=0.0)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--summary", action="store_true", help="omit the failure list")
    args = parser.parse_args()
    rows = load_rows(args.dataset)
    svc = build_service(args.docs, sorted({r["product"] for r in rows}))
    report = evaluate(svc, rows, k=args.k, use_product=not args.no_product)
    if args.summary:
        report.pop("failures")
    print(json.dumps(report, indent=2))
    if report[f"hit@{args.k}"] < args.min_hit or report["answer_accuracy"] < args.min_accuracy:
        print("Eval below threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

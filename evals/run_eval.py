"""Offline evaluation: retrieval hit rate, answer accuracy and citation precision.

    python -m evals.run_eval                         # in-memory store, current embedder/LLM settings
    python -m evals.run_eval --min-hit 0.9 --min-accuracy 0.7   # fail (exit 1) below thresholds

Metrics per question (dataset: evals/dataset.jsonl):
- hit@k:              the expected section is among the top-k retrieved chunks
- answer accuracy:    every expected term appears in the answer
- citation precision: share of cited sections that match the expected section
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import load_settings
from app.embeddings import HashingEmbedder
from app.schemas import AskRequest, IngestRequest
from app.service import DocService, _build_llm
from app.store import InMemoryStore

ROOT = Path(__file__).resolve().parent.parent


def build_service() -> DocService:
    settings = load_settings()
    svc = DocService(settings, InMemoryStore(), HashingEmbedder(settings.embedding_dim), _build_llm(settings))
    for product in ("payments", "notifications"):
        svc.ingest(IngestRequest(product=product, source=str(ROOT / "sample_docs" / product)))
    return svc


def evaluate(svc: DocService, k: int = 5, use_product: bool = True) -> dict:
    rows = [json.loads(line) for line in (ROOT / "evals" / "dataset.jsonl").read_text().splitlines() if line.strip()]
    hits = correct = 0
    precisions: list[float] = []
    failures = []
    for row in rows:
        product = row["product"] if use_product else None
        retrieved = svc.search(row["question"], k=k, product=product)
        hit = any(h.chunk.section.endswith(row["expected_section"]) for h in retrieved)
        resp = svc.ask(AskRequest(question=row["question"], product=product, top_k=k))
        ok = all(t.lower() in resp.answer.lower() for t in row["expected_terms"])
        if resp.citations:
            precisions.append(
                sum(c.section.endswith(row["expected_section"]) for c in resp.citations) / len(resp.citations)
            )
        else:
            precisions.append(0.0)
        hits += hit
        correct += ok
        if not (hit and ok):
            failures.append({"question": row["question"], "hit": hit, "answer_ok": ok, "answer": resp.answer[:160]})
    n = len(rows)
    return {
        "questions": n,
        "mode": "llm" if svc.llm else "extractive",
        f"hit@{k}": round(hits / n, 3),
        "answer_accuracy": round(correct / n, 3),
        "citation_precision": round(sum(precisions) / n, 3),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-product", action="store_true", help="do not pass the product hint")
    parser.add_argument("--min-hit", type=float, default=0.0)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    args = parser.parse_args()
    report = evaluate(build_service(), k=args.k, use_product=not args.no_product)
    print(json.dumps(report, indent=2))
    if report[f"hit@{args.k}"] < args.min_hit or report["answer_accuracy"] < args.min_accuracy:
        print("Eval below threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

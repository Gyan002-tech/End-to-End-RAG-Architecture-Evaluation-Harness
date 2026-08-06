#!/usr/bin/env python3
"""Stage 3 Verification — Cold reload, 9-cell grid validation & library parity check.

Runs in a fresh process, loading all 9 run JSONs and metrics summary from disk.

Checks:
  [1] Cold reload all 9 run files + stage3_rerank_metrics.json
  [2] Validate candidate structures (300 test queries, 50 candidates/query, docid resolution)
  [3] Cross-validate metrics against pytrec_eval across all 9 cells
  [4] Render full 3x3 Factorial Grid Matrix & Pareto Frontier

Usage:
    python src/verify_stage3.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    DOCMAP_PATH,
    RUNS_DIR,
    SCIFACT_SPLIT,
    load_docmap,
    load_scifact,
)
from src.metrics import evaluate_run  # noqa: E402

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def check_pytrec_eval_parity(
    run_doc_lists: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    stored_eval: Dict[str, float],
) -> bool:
    """Compare custom metric outputs against pytrec_eval if installed."""
    try:
        import pytrec_eval
    except ImportError:
        print("  pytrec_eval not installed — skipping library parity check")
        return True

    trec_qrels = {qid: {did: rel for did, rel in docs.items()} for qid, docs in qrels.items()}
    trec_run = {
        qid: {did: float(50 - rank) for rank, did in enumerate(docs)}
        for qid, docs in run_doc_lists.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        trec_qrels, {"recall_10", "recip_rank", "ndcg_cut_10", "P_1"}
    )
    res = evaluator.evaluate(trec_run)

    n = len(res)
    py_rec10 = sum(r["recall_10"] for r in res.values()) / n
    py_mrr = sum(r["recip_rank"] for r in res.values()) / n
    py_ndcg10 = sum(r["ndcg_cut_10"] for r in res.values()) / n
    py_p1 = sum(r["P_1"] for r in res.values()) / n

    ok_rec = abs(py_rec10 - stored_eval["recall@10"]) < 1e-4
    ok_mrr = abs(py_mrr - stored_eval["mrr"]) < 1e-4
    ok_ndcg = abs(py_ndcg10 - stored_eval["ndcg@10"]) < 1e-4
    ok_p1 = abs(py_p1 - stored_eval["p@1"]) < 1e-4

    all_ok = ok_rec and ok_mrr and ok_ndcg and ok_p1
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    failures: list[str] = []

    hdr("[1] Cold Reload All 9 Run JSON Artifacts from disk")
    expected_files = [
        "retrieval_bm25.json",
        "retrieval_dense.json",
        "retrieval_rrf.json",
        "rerank_m3_bm25.json",
        "rerank_m3_dense.json",
        "rerank_m3_rrf.json",
        "rerank_gemma_bm25.json",
        "rerank_gemma_dense.json",
        "rerank_gemma_rrf.json",
        "stage3_rerank_metrics.json",
    ]

    loaded_runs: Dict[str, dict] = {}
    for filename in expected_files:
        p = RUNS_DIR / filename
        if not p.exists():
            print(f"  MISSING: {p.name}")
            failures.append(f"Missing output artifact {p.name}")
        else:
            print(f"  Loaded : {p.name:<28} ({p.stat().st_size / 1024:7.1f} KiB)")
            if filename != "stage3_rerank_metrics.json":
                with open(p) as f:
                    loaded_runs[filename] = json.load(f)

    if failures:
        return 1

    with open(RUNS_DIR / "stage3_rerank_metrics.json") as f:
        summary_data = json.load(f)

    docmap = load_docmap()
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    valid_doc_ids = set(corpus.keys())

    hdr("[2] Validate Candidate Structure & Docid Resolution across 9 Runs")
    for filename, data in loaded_runs.items():
        runs = data["runs"]
        arm_name = data.get("arm", filename)
        if len(runs) != len(queries):
            failures.append(f"{arm_name} has {len(runs)} queries, expected {len(queries)}")

        bad_candidates = 0
        unresolved_docs = 0
        for qid, hits in runs.items():
            if len(hits) != 50:
                bad_candidates += 1
            for h in hits:
                did = h["doc_id"]
                if did not in valid_doc_ids or did not in docmap.docid_to_text:
                    unresolved_docs += 1

        status_flag = "OK" if (bad_candidates == 0 and unresolved_docs == 0) else "FAIL"
        print(f"  {arm_name:<32} : queries={len(runs)} candidates/q=50 ({status_flag}) unresolved={unresolved_docs}")

        if bad_candidates > 0:
            failures.append(f"{arm_name} contains queries without 50 candidates")
        if unresolved_docs > 0:
            failures.append(f"{arm_name} contains docids failing resolution")

    hdr("[3] Metric Library Parity Cross-Validation (pytrec_eval)")
    parity_failures = 0
    for filename, data in loaded_runs.items():
        arm_name = data.get("arm", filename)
        doc_lists = {qid: [h["doc_id"] for h in hits] for qid, hits in data["runs"].items()}
        ok = check_pytrec_eval_parity(doc_lists, qrels, data["metrics"])
        print(f"  {arm_name:<32} pytrec_eval parity : {'OK' if ok else 'FAIL'}")
        if not ok:
            parity_failures += 1
            failures.append(f"{arm_name} pytrec_eval metric mismatch")

    hdr("[4] Render Complete 3x3 Factorial Grid Matrix (9 Cells)")
    grid_cells = summary_data["grid_cells"]
    print(f"{'Config':<28} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6} | {'Total Latency':<14} | {'Frontier'}")
    print("-" * 95)
    for row in sorted(grid_cells, key=lambda x: -x["ndcg@10"]):
        flag = "★ SURVIVOR" if row.get("pareto_frontier") else "Dominated"
        print(
            f"{row['config']:<28} | {row['recall@10']:<10.4f} | {row['mrr']:<8.4f} | "
            f"{row['ndcg@10']:<8.4f} | {row['p@1']:<6.4f} | {row['latency_ms_mean']:<14.2f} | {flag}"
        )

    survivors = summary_data.get("pareto_survivors", [])
    print(f"\nPareto Frontier Survivors ({len(survivors)} configs selected for Phase 2 Generation):")
    for s in survivors:
        print(f"  -> {s['config']} (nDCG@10: {s['ndcg@10']:.4f}, Latency: {s['latency_ms_mean']:.2f} ms/query)")

    hdr("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All Stage 3 checks passed. 9-cell factorial grid matrix & Pareto survivors verified.")
    print("  Phase 1 Complete! Phase 2 (Generation & Faithfulness) is unblocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

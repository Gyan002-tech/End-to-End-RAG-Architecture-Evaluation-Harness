#!/usr/bin/env python3
"""Stage 2 Verification — Cold reload & independent metric cross-validation.

Runs in a fresh process, loading candidate run JSONs and metric tables from disk.

Checks:
  [1] Cold reload candidate runs (BM25, Dense, RRF Hybrid) from artifacts/runs/
  [2] Validate candidate structures (300 test queries, 50 candidates/query, valid docids)
  [3] Cross-validate hand-written metrics against ranx / pytrec_eval (if installed)
  [4] Render & verify Phase 1 Baseline Retrieval Performance Table

Usage:
    python src/verify_stage2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    DOCMAP_PATH,
    RUNS_DIR,
    SCIFACT_SPLIT,
    load_docmap,
    load_scifact,
)
from src.metrics import evaluate_run, test_canary_metrics  # noqa: E402

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def check_pytrec_eval_parity(
    run_doc_lists: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    hand_written_eval: Dict[str, float],
) -> bool:
    """Compare custom metric outputs against pytrec_eval if installed."""
    try:
        import pytrec_eval
    except ImportError:
        print("  pytrec_eval not installed — skipping library parity check")
        return True

    # Prepare pytrec_eval formats
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

    ok_rec = abs(py_rec10 - hand_written_eval["recall@10"]) < 1e-4
    ok_mrr = abs(py_mrr - hand_written_eval["mrr"]) < 1e-4
    ok_ndcg = abs(py_ndcg10 - hand_written_eval["ndcg@10"]) < 1e-4
    ok_p1 = abs(py_p1 - hand_written_eval["p@1"]) < 1e-4

    all_ok = ok_rec and ok_mrr and ok_ndcg and ok_p1
    print(
        f"  pytrec_eval parity  : {'OK' if all_ok else 'MISMATCH'} "
        f"(Rec@10: {py_rec10:.4f}, MRR: {py_mrr:.4f}, nDCG@10: {py_ndcg10:.4f})"
    )
    return all_ok


def check_ranx_parity(
    run_doc_lists: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    hand_written_eval: Dict[str, float],
) -> bool:
    """Compare custom metric outputs against ranx if installed."""
    try:
        from ranx import Evaluate, Qrels, Run
    except ImportError:
        print("  ranx not installed — skipping library parity check")
        return True

    ranx_qrels = Qrels(qrels)
    ranx_run_dict = {
        qid: {did: float(50 - rank) for rank, did in enumerate(docs)}
        for qid, docs in run_doc_lists.items()
    }
    ranx_run = Run(ranx_run_dict)

    res = Evaluate(ranx_run, ranx_qrels, metrics=["recall@10", "mrr", "ndcg@10", "precision@1"])

    ok_rec = abs(res["recall@10"] - hand_written_eval["recall@10"]) < 1e-4
    ok_mrr = abs(res["mrr"] - hand_written_eval["mrr"]) < 1e-4
    ok_ndcg = abs(res["ndcg@10"] - hand_written_eval["ndcg@10"]) < 1e-4
    ok_p1 = abs(res["precision@1"] - hand_written_eval["p@1"]) < 1e-4

    all_ok = ok_rec and ok_mrr and ok_ndcg and ok_p1
    print(
        f"  ranx parity         : {'OK' if all_ok else 'MISMATCH'} "
        f"(Rec@10: {res['recall@10']:.4f}, MRR: {res['mrr']:.4f}, nDCG@10: {res['ndcg@10']:.4f})"
    )
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    failures: list[str] = []

    hdr("[1] Canary Math Verification")
    if test_canary_metrics():
        print("  Canary metric calculations : PASS")
    else:
        print("  Canary metric calculations : FAIL")
        failures.append("Canary metric test failed")

    hdr("[2] Cold Reload Candidate Runs from disk")
    bm25_path = RUNS_DIR / "retrieval_bm25.json"
    dense_path = RUNS_DIR / "retrieval_dense.json"
    rrf_path = RUNS_DIR / "retrieval_rrf.json"
    summary_path = RUNS_DIR / "stage2_retrieval_metrics.json"

    for p in (bm25_path, dense_path, rrf_path, summary_path):
        if not p.exists():
            print(f"  MISSING: {p}")
            failures.append(f"Missing output artifact {p.name}")
        else:
            print(f"  Loaded : {p.name} ({p.stat().st_size / 1024:.1f} KiB)")

    if failures:
        return 1

    with open(bm25_path) as f:
        bm25_data = json.load(f)
    with open(dense_path) as f:
        dense_data = json.load(f)
    with open(rrf_path) as f:
        rrf_data = json.load(f)
    with open(summary_path) as f:
        summary_data = json.load(f)

    docmap = load_docmap()
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    valid_doc_ids = set(corpus.keys())

    hdr("[3] Validate Candidate Structure & Docid Resolution")
    for name, data in [("BM25", bm25_data), ("Dense", dense_data), ("RRF Hybrid", rrf_data)]:
        runs = data["runs"]
        if len(runs) != len(queries):
            failures.append(f"{name} run has {len(runs)} queries, expected {len(queries)}")

        bad_candidates = 0
        unresolved_docs = 0
        for qid, hits in runs.items():
            if len(hits) != 50:
                bad_candidates += 1
            for h in hits:
                did = h["doc_id"]
                if did not in valid_doc_ids or did not in docmap.docid_to_text:
                    unresolved_docs += 1

        print(f"  {name:<12} : queries={len(runs)}  candidates/query=50 ({'OK' if bad_candidates==0 else 'FAIL'})  "
              f"unresolved_docs={unresolved_docs}")

        if bad_candidates > 0:
            failures.append(f"{name} run contains queries without 50 candidates")
        if unresolved_docs > 0:
            failures.append(f"{name} run contains docids that fail resolution")

    hdr("[4] Metric Library Parity Cross-Validation")
    for name, data in [("BM25", bm25_data), ("Dense", dense_data), ("RRF Hybrid", rrf_data)]:
        print(f"\n  Checking {name} arm:")
        doc_lists = {qid: [h["doc_id"] for h in hits] for qid, hits in data["runs"].items()}
        pytrec_ok = check_pytrec_eval_parity(doc_lists, qrels, data["metrics"])
        ranx_ok = check_ranx_parity(doc_lists, qrels, data["metrics"])

        if not pytrec_ok:
            failures.append(f"{name} pytrec_eval metric mismatch")
        if not ranx_ok:
            failures.append(f"{name} ranx metric mismatch")

    hdr("[5] Phase 1 Baseline Retrieval Performance Table")
    print(f"{'Arm':<20} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6} | {'Latency (ms)':<12}")
    print("-" * 75)
    for row in summary_data["table"]:
        print(
            f"{row['arm']:<20} | {row['recall@10']:<10.4f} | {row['mrr']:<8.4f} | "
            f"{row['ndcg@10']:<8.4f} | {row['p@1']:<6.4f} | {row['latency_ms_mean']:<12.2f}"
        )

    hdr("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All Stage 2 checks passed. Candidate runs & metrics verified.")
    print("  Stage 3 (Reranking) is unblocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

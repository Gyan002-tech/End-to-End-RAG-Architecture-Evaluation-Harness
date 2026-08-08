#!/usr/bin/env python3
"""Stage 3 — Multi-Model Staged Reranking & 3x3 Factorial Grid Assembly.

Scores all 3 Stage 2 candidate sets (BM25, Dense, RRF Hybrid) through two rerankers:
  1. BAAI/bge-reranker-v2-m3 (Cross-Encoder)
  2. BAAI/bge-reranker-v2-gemma (2.5B LLM Reranker)

Enforces strict staged memory residency: loads ONE model at a time, scores all arms,
unloads from VRAM before loading the next.

IDEMPOTENCY & RESTARTABILITY:
  If a rerun occurs after a partial failure or timeout, individual completed
  runs (e.g. `rerank_m3_bm25.json`) are detected on disk and skipped automatically.
  Pass `--force` to force a complete re-run from scratch.

Outputs:
  - artifacts/runs/rerank_m3_bm25.json
  - artifacts/runs/rerank_m3_dense.json
  - artifacts/runs/rerank_m3_rrf.json
  - artifacts/runs/rerank_gemma_bm25.json
  - artifacts/runs/rerank_gemma_dense.json
  - artifacts/runs/rerank_gemma_rrf.json
  - artifacts/runs/stage3_rerank_metrics.json (3x3 grid & Pareto frontier)

Usage:
    python src/03_rerank.py
    python src/03_rerank.py --gemma-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.engine.common import (  # noqa: E402
    RUNS_DIR,
    SCIFACT_SPLIT,
    load_docmap,
    load_scifact,
)
from src.engine.metrics import evaluate_run  # noqa: E402
from src.engine.rerankers import GemmaReranker, M3Reranker, unload_model  # noqa: E402

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def rerank_candidate_set(
    reranker_obj: object,
    candidate_runs: Dict[str, List[dict]],
    queries: Dict[str, str],
    doc_texts: Dict[str, str],
    batch_size: int = 32,
) -> Tuple[Dict[str, List[dict]], Dict[str, List[str]], List[float]]:
    """Rerank candidates for all queries using the loaded reranker object."""
    reranked_runs: Dict[str, List[dict]] = {}
    reranked_doc_lists: Dict[str, List[str]] = {}
    latencies: List[float] = []

    qids = sorted(candidate_runs.keys(), key=lambda q: int(q) if q.isdigit() else q)

    for qid in qids:
        qtext = queries[qid]
        hits = candidate_runs[qid]

        pairs: List[Tuple[str, str]] = []
        candidate_docids: List[str] = []
        for h in hits:
            did = h["doc_id"]
            dtext = doc_texts.get(did, "")
            pairs.append((qtext, dtext))
            candidate_docids.append(did)

        st = time.perf_counter()
        scores = reranker_obj.compute_scores(pairs, batch_size=batch_size)
        dt = (time.perf_counter() - st) * 1000.0  # ms
        latencies.append(dt)

        # Pair docids with new reranker scores and sort descending
        scored_pairs = list(zip(candidate_docids, scores))
        sorted_pairs = sorted(scored_pairs, key=lambda x: -x[1])

        reranked_hits = []
        doc_list = []
        for rank, (did, sc) in enumerate(sorted_pairs, start=1):
            reranked_hits.append({"rank": rank, "doc_id": did, "score": float(sc)})
            doc_list.append(did)

        reranked_runs[qid] = reranked_hits
        reranked_doc_lists[qid] = doc_list

    return reranked_runs, reranked_doc_lists, latencies


def compute_pareto_frontier(grid_rows: List[dict]) -> List[dict]:
    """Identify Pareto non-dominated frontier configurations (Max nDCG@10 vs Min Latency)."""
    sorted_rows = sorted(grid_rows, key=lambda x: x["latency_ms_mean"])
    frontier: List[dict] = []
    max_ndcg = -1.0

    for row in sorted_rows:
        ndcg = row["ndcg@10"]
        if ndcg > max_ndcg:
            row["pareto_frontier"] = True
            frontier.append(row)
            max_ndcg = ndcg
        else:
            row["pareto_frontier"] = False

    return sorted_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=32, help="batch size for m3 reranker")
    ap.add_argument("--gemma-batch-size", type=int, default=4, help="batch size for gemma reranker")
    ap.add_argument("--gemma-4bit", action="store_true", help="force 4-bit bitsandbytes quantization for gemma")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="force re-running all reranker combinations")
    args = ap.parse_args()

    # Load candidate run files from Stage 2
    bm25_run_path = RUNS_DIR / "retrieval_bm25.json"
    dense_run_path = RUNS_DIR / "retrieval_dense.json"
    rrf_run_path = RUNS_DIR / "retrieval_rrf.json"
    summary_stage2_path = RUNS_DIR / "stage2_retrieval_metrics.json"

    for p in (bm25_run_path, dense_run_path, rrf_run_path, summary_stage2_path):
        if not p.exists():
            print(f"Missing required Stage 2 artifact: {p}")
            print("Run: python src/02_retrieve.py first.")
            return 1

    hdr("[1] Load Stage 2 Candidate Runs & SciFact Document Corpus")
    with open(bm25_run_path) as f:
        bm25_candidate_data = json.load(f)
    with open(dense_run_path) as f:
        dense_candidate_data = json.load(f)
    with open(rrf_run_path) as f:
        rrf_candidate_data = json.load(f)
    with open(summary_stage2_path) as f:
        stage2_summary = json.load(f)

    docmap = load_docmap()
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    doc_texts = docmap.docid_to_text

    candidate_arms = {
        "BM25": (bm25_candidate_data["runs"], stage2_summary["table"][0]["latency_ms_mean"]),
        "Dense": (dense_candidate_data["runs"], stage2_summary["table"][1]["latency_ms_mean"]),
        "RRF Hybrid": (rrf_candidate_data["runs"], stage2_summary["table"][2]["latency_ms_mean"]),
    }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    grid_rows: List[dict] = []

    # Include Stage 2 baselines in grid
    for row in stage2_summary["table"]:
        grid_rows.append({
            "retriever": row["arm"].split()[0],
            "reranker": "none",
            "config": f"{row['arm']} -> none",
            "recall@10": row["recall@10"],
            "mrr": row["mrr"],
            "ndcg@10": row["ndcg@10"],
            "p@1": row["p@1"],
            "retrieval_latency_ms": row["latency_ms_mean"],
            "rerank_latency_ms": 0.0,
            "latency_ms_mean": row["latency_ms_mean"],
        })

    # -----------------------------------------------------------------------
    # STAGE 3A: bge-reranker-v2-m3 (Cross-Encoder)
    # -----------------------------------------------------------------------
    hdr("[2] Stage 3A: Load bge-reranker-v2-m3 & Rerank Candidates")

    m3_model = None
    m3_arms_to_run = []
    for arm_name, (runs, ret_lat) in candidate_arms.items():
        file_slug = arm_name.lower().split()[0]
        out_path = RUNS_DIR / f"rerank_m3_{file_slug}.json"
        if out_path.exists() and not args.force:
            with open(out_path) as f:
                cached = json.load(f)
            eval_m = cached["metrics"]
            mean_rerank_lat = cached.get("rerank_latency_ms", 0.0)
            print(f"  [SKIPPED - CACHED] {out_path.name} (nDCG@10: {eval_m['ndcg@10']:.4f}, rerank: {mean_rerank_lat:.2f} ms/query)")
            grid_rows.append({
                "retriever": arm_name,
                "reranker": "bge-v2-m3",
                "config": f"{arm_name} -> bge-v2-m3",
                "recall@10": eval_m["recall@10"],
                "mrr": eval_m["mrr"],
                "ndcg@10": eval_m["ndcg@10"],
                "p@1": eval_m["p@1"],
                "retrieval_latency_ms": ret_lat,
                "rerank_latency_ms": round(mean_rerank_lat, 2),
                "latency_ms_mean": round(ret_lat + mean_rerank_lat, 2),
            })
        else:
            m3_arms_to_run.append((arm_name, runs, ret_lat, out_path))

    if m3_arms_to_run:
        print("  Loading BAAI/bge-reranker-v2-m3 onto CUDA...")
        m3_model = M3Reranker(device=device)

        for arm_name, runs, ret_lat, out_path in m3_arms_to_run:
            print(f"\n  Reranking {arm_name} candidates with bge-v2-m3...")
            t0 = time.perf_counter()
            reranked_runs, reranked_doc_lists, rerank_lats = rerank_candidate_set(
                m3_model, runs, queries, doc_texts, batch_size=args.batch_size
            )
            total_rerank_time = time.perf_counter() - t0
            mean_rerank_lat = float(np.mean(rerank_lats))

            eval_metrics = evaluate_run(reranked_doc_lists, qrels, k_list=(1, 5, 10, 20, 50))

            with open(out_path, "w") as f:
                json.dump({
                    "arm": f"{arm_name} -> bge-v2-m3",
                    "reranker": "BAAI/bge-reranker-v2-m3",
                    "runs": reranked_runs,
                    "metrics": eval_metrics,
                    "rerank_latency_ms": mean_rerank_lat,
                }, f, indent=2)
            print(f"    Wrote: {out_path.name} (nDCG@10: {eval_metrics['ndcg@10']:.4f}, rerank: {mean_rerank_lat:.2f} ms/query)")

            grid_rows.append({
                "retriever": arm_name,
                "reranker": "bge-v2-m3",
                "config": f"{arm_name} -> bge-v2-m3",
                "recall@10": eval_metrics["recall@10"],
                "mrr": eval_metrics["mrr"],
                "ndcg@10": eval_metrics["ndcg@10"],
                "p@1": eval_metrics["p@1"],
                "retrieval_latency_ms": ret_lat,
                "rerank_latency_ms": round(mean_rerank_lat, 2),
                "latency_ms_mean": round(ret_lat + mean_rerank_lat, 2),
            })

        print("\n  Unloading bge-v2-m3 model from GPU VRAM...")
        unload_model(m3_model)

    # -----------------------------------------------------------------------
    # STAGE 3B: bge-reranker-v2-gemma (2.5B LLM Reranker)
    # -----------------------------------------------------------------------
    hdr("[3] Stage 3B: Load bge-reranker-v2-gemma & Rerank Candidates")

    gemma_model = None
    gemma_arms_to_run = []
    gemma_quantized = args.gemma_4bit

    for arm_name, (runs, ret_lat) in candidate_arms.items():
        file_slug = arm_name.lower().split()[0]
        out_path = RUNS_DIR / f"rerank_gemma_{file_slug}.json"
        if out_path.exists() and not args.force:
            with open(out_path) as f:
                cached = json.load(f)
            eval_m = cached["metrics"]
            mean_rerank_lat = cached.get("rerank_latency_ms", 0.0)
            if cached.get("quantized_4bit"):
                gemma_quantized = True
            print(f"  [SKIPPED - CACHED] {out_path.name} (nDCG@10: {eval_m['ndcg@10']:.4f}, rerank: {mean_rerank_lat:.2f} ms/query)")
            grid_rows.append({
                "retriever": arm_name,
                "reranker": "bge-v2-gemma",
                "config": f"{arm_name} -> bge-v2-gemma",
                "recall@10": eval_m["recall@10"],
                "mrr": eval_m["mrr"],
                "ndcg@10": eval_m["ndcg@10"],
                "p@1": eval_m["p@1"],
                "retrieval_latency_ms": ret_lat,
                "rerank_latency_ms": round(mean_rerank_lat, 2),
                "latency_ms_mean": round(ret_lat + mean_rerank_lat, 2),
            })
        else:
            gemma_arms_to_run.append((arm_name, runs, ret_lat, out_path))

    if gemma_arms_to_run:
        print("  Loading BAAI/bge-reranker-v2-gemma onto CUDA...")
        gemma_bs = args.gemma_batch_size
        try:
            gemma_model = GemmaReranker(device=device, load_in_4bit=args.gemma_4bit)
        except Exception as exc:
            print(f"  fp16 load failed ({exc}) — falling back to 4-bit bitsandbytes quantization...")
            gemma_model = GemmaReranker(device=device, load_in_4bit=True)
            gemma_quantized = True

        for arm_name, runs, ret_lat, out_path in gemma_arms_to_run:
            print(f"\n  Reranking {arm_name} candidates with bge-v2-gemma (batch_size={gemma_bs})...")
            t0 = time.perf_counter()
            try:
                reranked_runs, reranked_doc_lists, rerank_lats = rerank_candidate_set(
                    gemma_model, runs, queries, doc_texts, batch_size=gemma_bs
                )
            except Exception as exc:
                if ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()) and not gemma_quantized:
                    print(f"\n  [VRAM OOM RECOVERY] Forward pass OOM in fp16. Unloading fp16 model and re-loading in 4-bit mode...")
                    unload_model(gemma_model)
                    gemma_model = GemmaReranker(device=device, load_in_4bit=True)
                    gemma_quantized = True
                    gemma_bs = min(gemma_bs, 4)
                    reranked_runs, reranked_doc_lists, rerank_lats = rerank_candidate_set(
                        gemma_model, runs, queries, doc_texts, batch_size=gemma_bs
                    )
                else:
                    raise

            total_rerank_time = time.perf_counter() - t0
            mean_rerank_lat = float(np.mean(rerank_lats))

            eval_metrics = evaluate_run(reranked_doc_lists, qrels, k_list=(1, 5, 10, 20, 50))

            with open(out_path, "w") as f:
                json.dump({
                    "arm": f"{arm_name} -> bge-v2-gemma",
                    "reranker": "BAAI/bge-reranker-v2-gemma",
                    "quantized_4bit": gemma_quantized,
                    "runs": reranked_runs,
                    "metrics": eval_metrics,
                    "rerank_latency_ms": mean_rerank_lat,
                }, f, indent=2)
            print(f"    Wrote: {out_path.name} (nDCG@10: {eval_metrics['ndcg@10']:.4f}, rerank: {mean_rerank_lat:.2f} ms/query)")

            grid_rows.append({
                "retriever": arm_name,
                "reranker": "bge-v2-gemma",
                "config": f"{arm_name} -> bge-v2-gemma",
                "recall@10": eval_metrics["recall@10"],
                "mrr": eval_metrics["mrr"],
                "ndcg@10": eval_metrics["ndcg@10"],
                "p@1": eval_metrics["p@1"],
                "retrieval_latency_ms": ret_lat,
                "rerank_latency_ms": round(mean_rerank_lat, 2),
                "latency_ms_mean": round(ret_lat + mean_rerank_lat, 2),
            })

        print("\n  Unloading bge-v2-gemma model from GPU VRAM...")
        unload_model(gemma_model)

    # -----------------------------------------------------------------------
    # STAGE 3C: 3x3 Factorial Grid Assembly & Pareto Frontier Analysis
    # -----------------------------------------------------------------------
    hdr("[4] Assemble 3x3 Factorial Grid Matrix & Pareto Frontier")

    analyzed_grid = compute_pareto_frontier(grid_rows)
    survivors = [row for row in analyzed_grid if row.get("pareto_frontier")]

    print("\nFull 3x3 Factorial Grid Matrix (9 Cells):")
    print(f"{'Config':<28} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6} | {'Total Latency':<14} | {'Frontier'}")
    print("-" * 95)
    for row in sorted(analyzed_grid, key=lambda x: -x["ndcg@10"]):
        flag = "★ SURVIVOR" if row.get("pareto_frontier") else "Dominated"
        print(
            f"{row['config']:<28} | {row['recall@10']:<10.4f} | {row['mrr']:<8.4f} | "
            f"{row['ndcg@10']:<8.4f} | {row['p@1']:<6.4f} | {row['latency_ms_mean']:<14.2f} | {flag}"
        )

    # Persist summary JSON
    summary_path = RUNS_DIR / "stage3_rerank_metrics.json"
    summary_data = {
        "schema_version": 1,
        "stage": "03_rerank",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gemma_quantized_4bit": gemma_quantized,
        "grid_cells": analyzed_grid,
        "pareto_survivors": survivors,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n  Wrote final summary: {summary_path.name}")

    hdr("Stage 3 complete")
    print(f"  3x3 Factorial Grid computed across 9 cells. Selected {len(survivors)} Pareto survivors.")
    print("  Next: python src/verify_stage3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

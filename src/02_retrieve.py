#!/usr/bin/env python3
"""Stage 2 — Phase 1 Candidate Retrieval & Metric Evaluation.

Executes candidate retrieval across all 300 SciFact test queries for 3 arms:
  1. BM25 (Sparse lexical baseline)
  2. Dense (BAAI/bge-base-en-v1.5 bi-encoder with max score chunk collapse)
  3. RRF Hybrid (Reciprocal Rank Fusion of BM25 + Dense, k=60)

Outputs candidate runs and metrics table to `artifacts/runs/`:
  - `retrieval_bm25.json`
  - `retrieval_dense.json`
  - `retrieval_rrf.json`
  - `stage2_retrieval_metrics.json`

Usage:
    python src/02_retrieve.py
    python src/02_retrieve.py --top-k 50 --rrf-k 60
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.common import (  # noqa: E402
    BGE_QUERY_INSTRUCTION,
    BM25_PATH,
    DOCMAP_PATH,
    FAISS_PATH,
    META_PATH,
    RUNS_DIR,
    SCIFACT_SPLIT,
    bm25_tokenize,
    embed_queries,
    load_bm25,
    load_docmap,
    load_embedder,
    load_faiss,
    load_meta,
    load_scifact,
    stage1_artifacts_present,
)
from src.metrics import evaluate_run, test_canary_metrics  # noqa: E402

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def reciprocal_rank_fusion(
    bm25_docids: List[str],
    dense_docids: List[str],
    rrf_k: int = 60,
    top_k: int = 50,
) -> List[Tuple[str, float]]:
    """RRF fusion on doc-level rank lists."""
    scores: Dict[str, float] = {}
    for rank, docid in enumerate(bm25_docids, start=1):
        scores[docid] = scores.get(docid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, docid in enumerate(dense_docids, start=1):
        scores[docid] = scores.get(docid, 0.0) + 1.0 / (rrf_k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_docs[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-k", type=int, default=50, help="number of top candidates to retrieve per query")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant k")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    args = ap.parse_args()

    if not stage1_artifacts_present():
        print("Stage 1 artifacts missing. Run: python src/01_index.py")
        return 1

    # Verify metric math upfront with canary vector
    if not test_canary_metrics():
        print("!! Metrics canary test failed. Aborting.")
        return 1

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hdr("[1] Load Stage 1 artifacts & SciFact test dataset")
    index = load_faiss()
    bm25_bundle = load_bm25()
    bm25 = bm25_bundle["bm25"]
    bm25_doc_ids = bm25_bundle["doc_ids"]
    docmap = load_docmap()
    meta = load_meta()
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)

    qids = sorted(queries.keys(), key=lambda q: int(q) if q.isdigit() else q)
    n_queries = len(qids)
    print(f"  test queries        : {n_queries}")
    print(f"  corpus docs         : {len(corpus)}")
    print(f"  docmap units        : {docmap.n_units} (dedup_needed={docmap.dedup_needed})")
    print(f"  FAISS ntotal        : {index.ntotal}")

    # Initialize embedder model for dense queries
    model = load_embedder(device=device)

    # -----------------------------------------------------------------------
    # Candidate Retrieval Loops
    # -----------------------------------------------------------------------
    hdr("[2] BM25 Candidate Retrieval (Sparse Arm)")
    bm25_runs: Dict[str, List[dict]] = {}
    bm25_doc_lists: Dict[str, List[str]] = {}
    bm25_latencies: List[float] = []

    t0 = time.perf_counter()
    for qid in qids:
        qtext = queries[qid]
        st = time.perf_counter()
        tokens = bm25_tokenize(qtext)
        scores = bm25.get_scores(tokens)
        top_idx = np.argsort(-scores)[: args.top_k]
        dt = (time.perf_counter() - st) * 1000.0  # ms
        bm25_latencies.append(dt)

        hits = []
        doc_list = []
        for rank, j in enumerate(top_idx, start=1):
            did = bm25_doc_ids[int(j)]
            sc = float(scores[j])
            hits.append({"rank": rank, "doc_id": did, "score": sc})
            doc_list.append(did)
        bm25_runs[qid] = hits
        bm25_doc_lists[qid] = doc_list
    t_bm25_total = time.perf_counter() - t0
    print(f"  BM25 retrieval total: {t_bm25_total:.2f}s ({np.mean(bm25_latencies):.2f} ms/query avg)")

    hdr("[3] Dense Candidate Retrieval (BAAI/bge-base-en-v1.5 + Max-Collapse)")
    # Embed queries in batch first (realistic production pattern), measure batch + search latencies
    query_texts = [queries[qid] for qid in qids]
    t0 = time.perf_counter()
    qvecs = embed_queries(model, query_texts, use_instruction=True, batch_size=64)
    t_dense_embed = time.perf_counter() - t0

    dense_runs: Dict[str, List[dict]] = {}
    dense_doc_lists: Dict[str, List[str]] = {}
    dense_latencies: List[float] = []

    # Search FAISS for top 200 units to guarantee at least top_k unique docs after collapse
    search_k = max(200, args.top_k * 4) if docmap.dedup_needed else args.top_k

    for idx, qid in enumerate(qids):
        st = time.perf_counter()
        single_qvec = qvecs[idx : idx + 1]
        scores, ordinals = index.search(single_qvec, search_k)

        # Chunk-to-Doc Max Score Collapse
        doc_scores: Dict[str, float] = {}
        for o, s in zip(ordinals[0], scores[0]):
            did = docmap.docid(int(o))
            sc = float(s)
            if did not in doc_scores or sc > doc_scores[did]:
                doc_scores[did] = sc

        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[: args.top_k]
        dt = (time.perf_counter() - st) * 1000.0  # ms
        dense_latencies.append(dt)

        hits = []
        doc_list = []
        for rank, (did, sc) in enumerate(sorted_docs, start=1):
            hits.append({"rank": rank, "doc_id": did, "score": sc})
            doc_list.append(did)
        dense_runs[qid] = hits
        dense_doc_lists[qid] = doc_list

    t_dense_search_total = sum(dense_latencies) / 1000.0
    embed_ms_per_q = (t_dense_embed * 1000.0) / n_queries
    search_ms_per_q = np.mean(dense_latencies)
    print(f"  Dense embed time    : {t_dense_embed:.2f}s ({embed_ms_per_q:.2f} ms/query)")
    print(f"  Dense FAISS search  : {t_dense_search_total:.2f}s ({search_ms_per_q:.2f} ms/query avg)")
    print(f"  Total Dense latency : {embed_ms_per_q + search_ms_per_q:.2f} ms/query avg")

    hdr("[4] RRF Hybrid Candidate Retrieval (BM25 + Dense Fusion, k=60)")
    rrf_runs: Dict[str, List[dict]] = {}
    rrf_doc_lists: Dict[str, List[str]] = {}
    rrf_latencies: List[float] = []

    for idx, qid in enumerate(qids):
        st = time.perf_counter()
        fused = reciprocal_rank_fusion(
            bm25_doc_lists[qid],
            dense_doc_lists[qid],
            rrf_k=args.rrf_k,
            top_k=args.top_k,
        )
        dt = (time.perf_counter() - st) * 1000.0
        rrf_latencies.append(dt)

        hits = []
        doc_list = []
        for rank, (did, sc) in enumerate(fused, start=1):
            hits.append({"rank": rank, "doc_id": did, "score": float(sc)})
            doc_list.append(did)
        rrf_runs[qid] = hits
        rrf_doc_lists[qid] = doc_list

    print(f"  RRF fusion total    : {sum(rrf_latencies):.2f} ms ({np.mean(rrf_latencies):.4f} ms/query avg)")

    # -----------------------------------------------------------------------
    # Evaluation Metrics & Comparison Table
    # -----------------------------------------------------------------------
    hdr("[5] Compute Hand-Written Retrieval Metrics & Latency Summary")

    bm25_eval = evaluate_run(bm25_doc_lists, qrels, k_list=(1, 5, 10, 20, 50))
    dense_eval = evaluate_run(dense_doc_lists, qrels, k_list=(1, 5, 10, 20, 50))
    rrf_eval = evaluate_run(rrf_doc_lists, qrels, k_list=(1, 5, 10, 20, 50))

    table_data = [
        {
            "arm": "BM25 (sparse)",
            "recall@10": bm25_eval["recall@10"],
            "mrr": bm25_eval["mrr"],
            "ndcg@10": bm25_eval["ndcg@10"],
            "p@1": bm25_eval["p@1"],
            "latency_ms_mean": round(float(np.mean(bm25_latencies)), 2),
            "latency_ms_p95": round(float(np.percentile(bm25_latencies, 95)), 2),
        },
        {
            "arm": "Dense (bge-base)",
            "recall@10": dense_eval["recall@10"],
            "mrr": dense_eval["mrr"],
            "ndcg@10": dense_eval["ndcg@10"],
            "p@1": dense_eval["p@1"],
            "latency_ms_mean": round(float(embed_ms_per_q + np.mean(dense_latencies)), 2),
            "latency_ms_p95": round(float(embed_ms_per_q + np.percentile(dense_latencies, 95)), 2),
        },
        {
            "arm": "RRF Hybrid",
            "recall@10": rrf_eval["recall@10"],
            "mrr": rrf_eval["mrr"],
            "ndcg@10": rrf_eval["ndcg@10"],
            "p@1": rrf_eval["p@1"],
            "latency_ms_mean": round(
                float(embed_ms_per_q + np.mean(bm25_latencies) + np.mean(dense_latencies) + np.mean(rrf_latencies)),
                2,
            ),
            "latency_ms_p95": round(
                float(
                    embed_ms_per_q
                    + np.percentile(bm25_latencies, 95)
                    + np.percentile(dense_latencies, 95)
                    + np.percentile(rrf_latencies, 95)
                ),
                2,
            ),
        },
    ]

    print("\nPhase 1 Retrieval Performance Baseline (3 Arms):")
    print(f"{'Arm':<20} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6} | {'Latency (ms)':<12}")
    print("-" * 75)
    for row in table_data:
        print(
            f"{row['arm']:<20} | {row['recall@10']:<10.4f} | {row['mrr']:<8.4f} | "
            f"{row['ndcg@10']:<8.4f} | {row['p@1']:<6.4f} | {row['latency_ms_mean']:<12.2f}"
        )

    # -----------------------------------------------------------------------
    # Persist JSON Artifacts
    # -----------------------------------------------------------------------
    hdr("[6] Persist Run Artifacts to artifacts/runs/")

    bm25_path = RUNS_DIR / "retrieval_bm25.json"
    dense_path = RUNS_DIR / "retrieval_dense.json"
    rrf_path = RUNS_DIR / "retrieval_rrf.json"
    metrics_path = RUNS_DIR / "stage2_retrieval_metrics.json"

    with open(bm25_path, "w") as f:
        json.dump({"arm": "bm25", "top_k": args.top_k, "runs": bm25_runs, "metrics": bm25_eval}, f, indent=2)

    with open(dense_path, "w") as f:
        json.dump({"arm": "dense", "top_k": args.top_k, "runs": dense_runs, "metrics": dense_eval}, f, indent=2)

    with open(rrf_path, "w") as f:
        json.dump(
            {"arm": "rrf_hybrid", "top_k": args.top_k, "rrf_k": args.rrf_k, "runs": rrf_runs, "metrics": rrf_eval},
            f,
            indent=2,
        )

    summary = {
        "schema_version": 1,
        "stage": "02_retrieve",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table": table_data,
        "metrics_detail": {
            "bm25": bm25_eval,
            "dense": dense_eval,
            "rrf_hybrid": rrf_eval,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Wrote: {bm25_path.name}")
    print(f"  Wrote: {dense_path.name}")
    print(f"  Wrote: {rrf_path.name}")
    print(f"  Wrote: {metrics_path.name}")

    hdr("Stage 2 complete")
    print("  Candidate runs & evaluation summary persisted cleanly.")
    print("  Next: python src/verify_stage2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 2c — Candidate Retrieval & 3-Tier Model Size Sweet-Spot Evaluation.

Evaluates `BAAI/bge-small-en-v1.5` (33M params, 384-dim) dense retrieval across 300 test queries.

Assembles the full 3-Tier Model Size Benchmark Matrix:
  1. Small: BAAI/bge-small-en-v1.5 (33M params, 384-dim)
  2. Base:  BAAI/bge-base-en-v1.5  (109M params, 768-dim)
  3. Large: intfloat/e5-large-v2   (335M params, 1024-dim)

Outputs:
  - artifacts/runs/retrieval_bge_small_dense.json
  - artifacts/runs/dense_model_size_sweetspot.json

Usage:
    python src/02c_retrieve_bge_small.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.common import (  # noqa: E402
    INDEX_DIR,
    RUNS_DIR,
    SCIFACT_SPLIT,
    embed_queries,
    load_scifact,
)
from src.metrics import evaluate_run  # noqa: E402

BANNER = "=" * 74
BGE_SMALL_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_SMALL_PATH = INDEX_DIR / "faiss_bge_small.index"
DOCMAP_SMALL_PATH = INDEX_DIR / "docmap_bge_small.json"


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    if not FAISS_SMALL_PATH.exists() or not DOCMAP_SMALL_PATH.exists():
        print("Stage 1c (bge-small) artifacts missing. Run: python src/01c_index_bge_small.py")
        return 1

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hdr("[1] Load bge-small Index & SciFact Test Queries")
    index = faiss.read_index(str(FAISS_SMALL_PATH))
    with open(DOCMAP_SMALL_PATH) as f:
        docmap = json.load(f)

    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    qids = sorted(queries.keys(), key=lambda q: int(q) if q.isdigit() else q)
    doc_ids = docmap["ordinal_to_docid"]

    print(f"  bge-small FAISS ntotal : {index.ntotal} (dim: {index.d})")
    print(f"  test queries           : {len(qids)}")

    hdr("[2] Embed Queries with BGE Instruction & Retrieve Top-50")
    model_kwargs = {"attn_implementation": "sdpa"}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    embedder = SentenceTransformer(BGE_SMALL_MODEL, device=device, model_kwargs=model_kwargs)
    embedder.max_seq_length = 512

    query_texts = [queries[q] for q in qids]

    t0 = time.perf_counter()
    raw_qvecs = embed_queries(embedder, query_texts, use_instruction=True, batch_size=64)
    t_embed = time.perf_counter() - t0
    embed_ms_per_q = (t_embed * 1000.0) / len(qids)

    qvecs = np.ascontiguousarray(raw_qvecs, dtype=np.float32)

    small_runs: Dict[str, List[dict]] = {}
    small_doc_lists: Dict[str, List[str]] = {}
    search_latencies: List[float] = []

    for idx, qid in enumerate(qids):
        st = time.perf_counter()
        single_vec = qvecs[idx : idx + 1]
        scores, ordinals = index.search(single_vec, args.top_k)
        dt = (time.perf_counter() - st) * 1000.0
        search_latencies.append(dt)

        hits = []
        doc_list = []
        for rank, (o, s) in enumerate(zip(ordinals[0], scores[0]), start=1):
            did = doc_ids[int(o)]
            sc = float(s)
            hits.append({"rank": rank, "doc_id": did, "score": sc})
            doc_list.append(did)
        small_runs[qid] = hits
        small_doc_lists[qid] = doc_list

    search_ms_per_q = float(np.mean(search_latencies))
    total_lat_ms = embed_ms_per_q + search_ms_per_q

    hdr("[3] Compute bge-small Metrics & Assemble 3-Tier Sweet-Spot Matrix")
    small_eval = evaluate_run(small_doc_lists, qrels, k_list=(1, 5, 10, 20, 50))

    # Load bge-base (Stage 2) and e5-large (Stage 2b) metrics
    stage2_path = RUNS_DIR / "stage2_retrieval_metrics.json"
    stage2b_path = RUNS_DIR / "e5_vs_bge_comparison.json"

    base_eval = {}
    large_eval = {}
    large_lat = 0.0

    if stage2_path.exists():
        with open(stage2_path) as f:
            s2 = json.load(f)
        base_eval = s2["metrics_detail"].get("dense", {})

    if stage2b_path.exists():
        with open(stage2b_path) as f:
            s2b = json.load(f)
        large_eval = s2b.get("e5_large_v2", {})
        large_lat = s2b.get("e5_latency_ms", 0.0)

    matrix = [
        {
            "tier": "Small (33M params)",
            "model": "BAAI/bge-small-en-v1.5",
            "dim": 384,
            "params": "33M",
            "recall@10": small_eval["recall@10"],
            "mrr": small_eval["mrr"],
            "ndcg@10": small_eval["ndcg@10"],
            "p@1": small_eval["p@1"],
            "latency_ms": round(total_lat_ms, 2),
        },
        {
            "tier": "Base (109M params) ★",
            "model": "BAAI/bge-base-en-v1.5",
            "dim": 768,
            "params": "109M",
            "recall@10": base_eval.get("recall@10", 0.0),
            "mrr": base_eval.get("mrr", 0.0),
            "ndcg@10": base_eval.get("ndcg@10", 0.0),
            "p@1": base_eval.get("p@1", 0.0),
            "latency_ms": 3.19,
        },
        {
            "tier": "Large (335M params)",
            "model": "intfloat/e5-large-v2",
            "dim": 1024,
            "params": "335M",
            "recall@10": large_eval.get("recall@10", 0.0),
            "mrr": large_eval.get("mrr", 0.0),
            "ndcg@10": large_eval.get("ndcg@10", 0.0),
            "p@1": large_eval.get("p@1", 0.0),
            "latency_ms": large_lat,
        },
    ]

    print("\n3-Tier Dense Embedding Model Size Benchmark (Small vs Base vs Large):")
    print(f"{'Tier':<22} | {'Model':<24} | {'Dim':<5} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6}")
    print("-" * 95)
    for row in matrix:
        print(
            f"{row['tier']:<22} | {row['model']:<24} | {row['dim']:<5} | "
            f"{row['recall@10']:<10.4f} | {row['mrr']:<8.4f} | {row['ndcg@10']:<8.4f} | {row['p@1']:<6.4f}"
        )

    # Persist outputs
    out_run_path = RUNS_DIR / "retrieval_bge_small_dense.json"
    with open(out_run_path, "w") as f:
        json.dump({"arm": "bge_small_v1_5", "top_k": args.top_k, "runs": small_runs, "metrics": small_eval}, f, indent=2)

    out_matrix_path = RUNS_DIR / "dense_model_size_sweetspot.json"
    summary = {
        "schema_version": 1,
        "stage": "02c_retrieve_bge_small",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sweetspot_matrix": matrix,
    }
    with open(out_matrix_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Wrote: {out_run_path.name}")
    print(f"  Wrote: {out_matrix_path.name}")
    hdr("Stage 2c complete")
    print("  Next: python src/verify_bge_small.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

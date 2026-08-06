#!/usr/bin/env python3
"""Stage 2b — Candidate Retrieval & Metric Evaluation for 2nd Dense Model (e5-large-v2).

Evaluates `intfloat/e5-large-v2` dense retrieval across all 300 SciFact test queries,
prepending `query: ` prefix to queries per e5 model specification.

Compares initial retrieval performance directly against `BAAI/bge-base-en-v1.5`.

Outputs:
  - artifacts/runs/retrieval_e5_dense.json
  - artifacts/runs/e5_vs_bge_comparison.json

Usage:
    python src/02b_retrieve_e5.py
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
    load_scifact,
)
from src.metrics import evaluate_run  # noqa: E402

BANNER = "=" * 74
E5_MODEL = "intfloat/e5-large-v2"
FAISS_E5_PATH = INDEX_DIR / "faiss_e5.index"
DOCMAP_E5_PATH = INDEX_DIR / "docmap_e5.json"


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    if not FAISS_E5_PATH.exists() or not DOCMAP_E5_PATH.exists():
        print("Stage 1b (e5) artifacts missing. Run: python src/01b_index_e5.py")
        return 1

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hdr("[1] Load e5 Index & SciFact Test Queries")
    index = faiss.read_index(str(FAISS_E5_PATH))
    with open(DOCMAP_E5_PATH) as f:
        docmap = json.load(f)

    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    qids = sorted(queries.keys(), key=lambda q: int(q) if q.isdigit() else q)
    doc_ids = docmap["ordinal_to_docid"]

    print(f"  e5 FAISS ntotal     : {index.ntotal} (dim: {index.d})")
    print(f"  test queries        : {len(qids)}")

    hdr("[2] Encode Queries with 'query: ' Prefix & Retrieve Top-50")
    model_kwargs = {"attn_implementation": "sdpa"}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    embedder = SentenceTransformer(E5_MODEL, device=device, model_kwargs=model_kwargs)
    embedder.max_seq_length = 512

    # e5 models require 'query: ' prefix for queries
    prefixed_queries = [f"query: {queries[q]}" for q in qids]

    t0 = time.perf_counter()
    raw_qvecs = embedder.encode(
        prefixed_queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    t_embed = time.perf_counter() - t0
    embed_ms_per_q = (t_embed * 1000.0) / len(qids)

    qvecs = np.ascontiguousarray(raw_qvecs, dtype=np.float32)

    e5_runs: Dict[str, List[dict]] = {}
    e5_doc_lists: Dict[str, List[str]] = {}
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
        e5_runs[qid] = hits
        e5_doc_lists[qid] = doc_list

    search_ms_per_q = float(np.mean(search_latencies))
    total_lat_ms = embed_ms_per_q + search_ms_per_q

    hdr("[3] Evaluate e5-large-v2 Metrics & Compare vs bge-base-en-v1.5")
    e5_eval = evaluate_run(e5_doc_lists, qrels, k_list=(1, 5, 10, 20, 50))

    # Load Stage 2 bge-base metrics for direct side-by-side benchmarking
    stage2_path = RUNS_DIR / "stage2_retrieval_metrics.json"
    bge_eval = {}
    if stage2_path.exists():
        with open(stage2_path) as f:
            s2 = json.load(f)
        bge_eval = s2["metrics_detail"].get("dense", {})

    print("\nDense Embedding Model Comparison Baseline (SciFact Test Set):")
    print(f"{'Model':<28} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6} | {'Latency (ms)'}")
    print("-" * 85)
    print(
        f"{'BAAI/bge-base-en-v1.5':<28} | {bge_eval.get('recall@10', 0.0):<10.4f} | "
        f"{bge_eval.get('mrr', 0.0):<8.4f} | {bge_eval.get('ndcg@10', 0.0):<8.4f} | "
        f"{bge_eval.get('p@1', 0.0):<6.4f} | {'3.19 ms'}"
    )
    print(
        f"{'intfloat/e5-large-v2':<28} | {e5_eval['recall@10']:<10.4f} | "
        f"{e5_eval['mrr']:<8.4f} | {e5_eval['ndcg@10']:<8.4f} | "
        f"{e5_eval['p@1']:<6.4f} | {total_lat_ms:<10.2f} ms"
    )

    # Persist outputs
    out_run_path = RUNS_DIR / "retrieval_e5_dense.json"
    with open(out_run_path, "w") as f:
        json.dump({"arm": "e5_large_v2", "top_k": args.top_k, "runs": e5_runs, "metrics": e5_eval}, f, indent=2)

    out_cmp_path = RUNS_DIR / "e5_vs_bge_comparison.json"
    cmp_summary = {
        "schema_version": 1,
        "stage": "02b_retrieve_e5",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bge_base_v1_5": bge_eval,
        "e5_large_v2": e5_eval,
        "e5_latency_ms": round(total_lat_ms, 2),
    }
    with open(out_cmp_path, "w") as f:
        json.dump(cmp_summary, f, indent=2)

    print(f"\n  Wrote: {out_run_path.name}")
    print(f"  Wrote: {out_cmp_path.name}")
    hdr("Stage 2b complete")
    print("  Next: python src/verify_e5.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

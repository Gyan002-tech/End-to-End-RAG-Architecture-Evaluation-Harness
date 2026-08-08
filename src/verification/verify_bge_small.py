#!/usr/bin/env python3
"""Stage 1c/2c Verification — Cold reload & 3-Tier Dense Model Size Sweet-Spot Check.

Usage:
    python src/verify_bge_small.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.engine.common import INDEX_DIR, RUNS_DIR, SCIFACT_SPLIT, load_scifact

BANNER = "=" * 74
FAISS_SMALL_PATH = INDEX_DIR / "faiss_bge_small.index"
DOCMAP_SMALL_PATH = INDEX_DIR / "docmap_bge_small.json"
META_SMALL_PATH = INDEX_DIR / "index_meta_bge_small.json"
RUN_SMALL_PATH = RUNS_DIR / "retrieval_bge_small_dense.json"
MATRIX_PATH = RUNS_DIR / "dense_model_size_sweetspot.json"


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    failures: list[str] = []

    hdr("[1] Cold Reload bge-small Artifacts from disk")
    for p in (FAISS_SMALL_PATH, DOCMAP_SMALL_PATH, META_SMALL_PATH, RUN_SMALL_PATH, MATRIX_PATH):
        if not p.exists():
            print(f"  MISSING: {p.name}")
            failures.append(f"Missing artifact {p.name}")
        else:
            print(f"  Loaded : {p.name:<30} ({p.stat().st_size / 1024:7.1f} KiB)")

    if failures:
        return 1

    import faiss

    index = faiss.read_index(str(FAISS_SMALL_PATH))
    with open(DOCMAP_SMALL_PATH) as f:
        docmap = json.load(f)
    with open(RUN_SMALL_PATH) as f:
        run_data = json.load(f)
    with open(MATRIX_PATH) as f:
        matrix_data = json.load(f)

    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)

    hdr("[2] FAISS & Docmap Assertions (bge-small-en-v1.5)")
    print(f"  index.ntotal        : {index.ntotal} (expect 5183)")
    print(f"  index.d             : {index.d} (expect 384 for bge-small)")

    if index.ntotal != len(corpus):
        failures.append(f"ntotal {index.ntotal} != corpus docs {len(corpus)}")
    if index.d != 384:
        failures.append(f"index.d {index.d} != 384")

    hdr("[3] Candidate Run Structure & Docid Resolution")
    runs = run_data["runs"]
    print(f"  queries in run      : {len(runs)}")

    bad_candidates = 0
    unresolved_docs = 0
    valid_docids = set(corpus.keys())

    for qid, hits in runs.items():
        if len(hits) != 50:
            bad_candidates += 1
        for h in hits:
            if h["doc_id"] not in valid_docids:
                unresolved_docs += 1

    print(f"  candidates/query=50 : {'OK' if bad_candidates == 0 else 'FAIL'}")
    print(f"  unresolved docids   : {unresolved_docs}")

    if bad_candidates > 0:
        failures.append("Run contains queries without 50 candidates")
    if unresolved_docs > 0:
        failures.append("Run contains unresolved docids")

    hdr("[4] Render 3-Tier Dense Model Size Sweet-Spot Benchmark")
    matrix = matrix_data.get("sweetspot_matrix", [])

    print(f"{'Tier':<22} | {'Model':<24} | {'Dim':<5} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6}")
    print("-" * 95)
    for row in matrix:
        print(
            f"{row['tier']:<22} | {row['model']:<24} | {row['dim']:<5} | "
            f"{row['recall@10']:<10.4f} | {row['mrr']:<8.4f} | {row['ndcg@10']:<8.4f} | {row['p@1']:<6.4f}"
        )

    hdr("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All bge-small dense embedding checks passed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

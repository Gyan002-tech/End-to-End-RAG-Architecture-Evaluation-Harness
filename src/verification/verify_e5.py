#!/usr/bin/env python3
"""Stage 1b/2b Verification — Cold reload & library parity check for e5-large-v2.

Usage:
    python src/verify_e5.py
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
FAISS_E5_PATH = INDEX_DIR / "faiss_e5.index"
DOCMAP_E5_PATH = INDEX_DIR / "docmap_e5.json"
META_E5_PATH = INDEX_DIR / "index_meta_e5.json"
RUN_E5_PATH = RUNS_DIR / "retrieval_e5_dense.json"
CMP_E5_PATH = RUNS_DIR / "e5_vs_bge_comparison.json"


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    failures: list[str] = []

    hdr("[1] Cold Reload e5 Artifacts from disk")
    for p in (FAISS_E5_PATH, DOCMAP_E5_PATH, META_E5_PATH, RUN_E5_PATH, CMP_E5_PATH):
        if not p.exists():
            print(f"  MISSING: {p.name}")
            failures.append(f"Missing artifact {p.name}")
        else:
            print(f"  Loaded : {p.name:<24} ({p.stat().st_size / 1024:7.1f} KiB)")

    if failures:
        return 1

    import faiss

    index = faiss.read_index(str(FAISS_E5_PATH))
    with open(DOCMAP_E5_PATH) as f:
        docmap = json.load(f)
    with open(RUN_E5_PATH) as f:
        run_data = json.load(f)
    with open(CMP_E5_PATH) as f:
        cmp_data = json.load(f)

    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)

    hdr("[2] FAISS & Docmap Assertions (e5-large-v2)")
    print(f"  index.ntotal        : {index.ntotal} (expect 5183)")
    print(f"  index.d             : {index.d} (expect 1024 for e5-large)")

    if index.ntotal != len(corpus):
        failures.append(f"ntotal {index.ntotal} != corpus docs {len(corpus)}")
    if index.d != 1024:
        failures.append(f"index.d {index.d} != 1024")

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

    hdr("[4] Dense Embedding Model Comparison Baseline")
    bge_m = cmp_data.get("bge_base_v1_5", {})
    e5_m = cmp_data.get("e5_large_v2", {})

    print(f"{'Model':<28} | {'Recall@10':<10} | {'MRR':<8} | {'nDCG@10':<8} | {'P@1':<6}")
    print("-" * 72)
    print(
        f"{'BAAI/bge-base-en-v1.5':<28} | {bge_m.get('recall@10', 0.0):<10.4f} | "
        f"{bge_m.get('mrr', 0.0):<8.4f} | {bge_m.get('ndcg@10', 0.0):<8.4f} | {bge_m.get('p@1', 0.0):<6.4f}"
    )
    print(
        f"{'intfloat/e5-large-v2':<28} | {e5_m.get('recall@10', 0.0):<10.4f} | "
        f"{e5_m.get('mrr', 0.0):<8.4f} | {e5_m.get('ndcg@10', 0.0):<8.4f} | {e5_m.get('p@1', 0.0):<6.4f}"
    )

    hdr("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All e5 dense embedding checks passed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

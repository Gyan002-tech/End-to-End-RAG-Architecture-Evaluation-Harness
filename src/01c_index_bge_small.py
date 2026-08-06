#!/usr/bin/env python3
"""Stage 1c — Build and persist Small Dense Retrieval Index (BAAI/bge-small-en-v1.5).

Small dense model arm for parameter sweet-spot benchmark (33M params, 384-dim).

Outputs:
  - artifacts/index/faiss_bge_small.index (384-dim FAISS IndexFlatIP)
  - artifacts/index/docmap_bge_small.json
  - artifacts/index/index_meta_bge_small.json

Usage:
    python src/01c_index_bge_small.py
    python src/01c_index_bge_small.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.common import (  # noqa: E402
    ATTN_IMPLEMENTATION,
    DEFAULT_BATCH_SIZE,
    INDEX_DIR,
    MAX_SEQ_LENGTH,
    SCIFACT_SPLIT,
    doc_text,
    embed_passages,
    load_embedder,
    load_scifact,
    package_versions,
    sorted_doc_ids,
)

BANNER = "=" * 74
BGE_SMALL_MODEL = "BAAI/bge-small-en-v1.5"
BGE_SMALL_DIM = 384

FAISS_SMALL_PATH = INDEX_DIR / "faiss_bge_small.index"
DOCMAP_SMALL_PATH = INDEX_DIR / "docmap_bge_small.json"
META_SMALL_PATH = INDEX_DIR / "index_meta_bge_small.json"


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if FAISS_SMALL_PATH.exists() and DOCMAP_SMALL_PATH.exists() and META_SMALL_PATH.exists() and not args.force:
        hdr("Stage 1c (bge-small) — SKIPPED (artifacts already present)")
        print(f"  {FAISS_SMALL_PATH.name} ({FAISS_SMALL_PATH.stat().st_size / 1024:.0f} KiB)")
        print(f"  {DOCMAP_SMALL_PATH.name} ({DOCMAP_SMALL_PATH.stat().st_size / 1024:.0f} KiB)")
        print("\n  Pass --force to rebuild. Nothing was changed.")
        return 0

    t_start = time.perf_counter()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hdr("[1] Load SciFact Corpus for bge-small-en-v1.5 Indexing")
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    doc_ids = sorted_doc_ids(corpus)
    texts = [doc_text(corpus[d]) for d in doc_ids]

    hdr("[2] Load BAAI/bge-small-en-v1.5 & Embed Passages (fp16 on CUDA)")
    print(f"  model               : {BGE_SMALL_MODEL}")
    print(f"  dim                 : {BGE_SMALL_DIM}")
    print(f"  parameters          : ~33M")

    model_kwargs = {"attn_implementation": ATTN_IMPLEMENTATION}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    t0 = time.perf_counter()
    embedder = SentenceTransformer(BGE_SMALL_MODEL, device=device, model_kwargs=model_kwargs)
    embedder.max_seq_length = MAX_SEQ_LENGTH
    t_model_load = time.perf_counter() - t0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    vectors = embed_passages(embedder, texts, batch_size=args.batch_size, progress=True)
    t_embed = time.perf_counter() - t0

    peak_alloc = peak_reserved = None
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    print(f"\n  model load time     : {t_model_load:.1f}s")
    print(f"  embed time          : {t_embed:.1f}s")
    print(f"  vector shape        : {vectors.shape}")

    hdr("[3] Build + Persist FAISS IndexFlatIP (bge-small-en-v1.5)")
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(BGE_SMALL_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_SMALL_PATH))
    t_faiss = time.perf_counter() - t0

    docmap_data = {
        "schema_version": 1,
        "model": BGE_SMALL_MODEL,
        "dim": BGE_SMALL_DIM,
        "n_units": len(doc_ids),
        "n_docs": len(doc_ids),
        "ordinal_to_docid": doc_ids,
        "docid_to_ordinals": {d: [i] for i, d in enumerate(doc_ids)},
        "docid_to_text": {d: t for d, t in zip(doc_ids, texts)},
    }
    with open(DOCMAP_SMALL_PATH, "w") as f:
        json.dump(docmap_data, f)

    t_total = time.perf_counter() - t_start
    meta_data = {
        "schema_version": 1,
        "stage": "01c_index_bge_small",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": BGE_SMALL_MODEL,
        "dim": BGE_SMALL_DIM,
        "params": "33M",
        "faiss_ntotal": index.ntotal,
        "timings_sec": {
            "model_load": round(t_model_load, 2),
            "embed": round(t_embed, 2),
            "faiss_write": round(t_faiss, 2),
            "total": round(t_total, 2),
        },
        "versions": package_versions(),
    }
    with open(META_SMALL_PATH, "w") as f:
        json.dump(meta_data, f, indent=2)

    hdr("Stage 1c (bge-small) complete")
    print(f"  FAISS index ntotal : {index.ntotal}  dim: {index.d}")
    print(f"  Persisted          : {FAISS_SMALL_PATH.name}")
    print(f"  Persisted          : {DOCMAP_SMALL_PATH.name}")
    print(f"  Persisted          : {META_SMALL_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 1b — Build and persist 2nd Dense Retrieval Index (intfloat/e5-large-v2).

Second dense model arm for benchmark completeness per methodology §2.
Model: `intfloat/e5-large-v2` (1024-dim, requires `passage: ` / `query: ` prefixes).

Outputs:
  - artifacts/index/faiss_e5.index (1024-dim FAISS IndexFlatIP)
  - artifacts/index/docmap_e5.json
  - artifacts/index/index_meta_e5.json

Usage:
    python src/01b_index_e5.py
    python src/01b_index_e5.py --force
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
    SCIFACT_URL,
    doc_text,
    load_scifact,
    package_versions,
    sorted_doc_ids,
)

BANNER = "=" * 74
E5_MODEL = "intfloat/e5-large-v2"
E5_DIM = 1024

FAISS_E5_PATH = INDEX_DIR / "faiss_e5.index"
DOCMAP_E5_PATH = INDEX_DIR / "docmap_e5.json"
META_E5_PATH = INDEX_DIR / "index_meta_e5.json"


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

    if FAISS_E5_PATH.exists() and DOCMAP_E5_PATH.exists() and META_E5_PATH.exists() and not args.force:
        hdr("Stage 1b (e5-large-v2) — SKIPPED (artifacts already present)")
        print(f"  {FAISS_E5_PATH.name} ({FAISS_E5_PATH.stat().st_size / 1024:.0f} KiB)")
        print(f"  {DOCMAP_E5_PATH.name} ({DOCMAP_E5_PATH.stat().st_size / 1024:.0f} KiB)")
        print("\n  Pass --force to rebuild. Nothing was changed.")
        return 0

    t_start = time.perf_counter()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hdr("[1] Load SciFact Corpus for e5-large-v2 Indexing")
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    doc_ids = sorted_doc_ids(corpus)
    texts = [doc_text(corpus[d]) for d in doc_ids]

    # e5 models require 'passage: ' prefix for passages
    prefixed_passages = [f"passage: {t}" for t in texts]

    hdr("[2] Load intfloat/e5-large-v2 & Embed Passages (fp16 on CUDA)")
    print(f"  model               : {E5_MODEL}")
    print(f"  dim                 : {E5_DIM}")
    print(f"  passage prefix      : 'passage: '")

    model_kwargs = {"attn_implementation": ATTN_IMPLEMENTATION}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    t0 = time.perf_counter()
    embedder = SentenceTransformer(E5_MODEL, device=device, model_kwargs=model_kwargs)
    embedder.max_seq_length = MAX_SEQ_LENGTH
    t_model_load = time.perf_counter() - t0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    raw_vectors = embedder.encode(
        prefixed_passages,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    t_embed = time.perf_counter() - t0

    vectors = np.ascontiguousarray(raw_vectors, dtype=np.float32)

    peak_alloc = peak_reserved = None
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    print(f"\n  model load time     : {t_model_load:.1f}s")
    print(f"  embed time          : {t_embed:.1f}s")
    print(f"  vector shape        : {vectors.shape}")

    hdr("[3] Build + Persist FAISS IndexFlatIP (e5-large-v2)")
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(E5_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_E5_PATH))
    t_faiss = time.perf_counter() - t0

    docmap_data = {
        "schema_version": 1,
        "model": E5_MODEL,
        "dim": E5_DIM,
        "n_units": len(doc_ids),
        "n_docs": len(doc_ids),
        "ordinal_to_docid": doc_ids,
        "docid_to_ordinals": {d: [i] for i, d in enumerate(doc_ids)},
        "docid_to_text": {d: t for d, t in zip(doc_ids, texts)},
    }
    with open(DOCMAP_E5_PATH, "w") as f:
        json.dump(docmap_data, f)

    t_total = time.perf_counter() - t_start
    meta_data = {
        "schema_version": 1,
        "stage": "01b_index_e5",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": E5_MODEL,
        "dim": E5_DIM,
        "prefix_passage": "passage: ",
        "prefix_query": "query: ",
        "faiss_ntotal": index.ntotal,
        "timings_sec": {
            "model_load": round(t_model_load, 2),
            "embed": round(t_embed, 2),
            "faiss_write": round(t_faiss, 2),
            "total": round(t_total, 2),
        },
        "versions": package_versions(),
    }
    with open(META_E5_PATH, "w") as f:
        json.dump(meta_data, f, indent=2)

    hdr("Stage 1b (e5-large-v2) complete")
    print(f"  FAISS index ntotal : {index.ntotal}  dim: {index.d}")
    print(f"  Persisted          : {FAISS_E5_PATH.name}")
    print(f"  Persisted          : {DOCMAP_E5_PATH.name}")
    print(f"  Persisted          : {META_E5_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

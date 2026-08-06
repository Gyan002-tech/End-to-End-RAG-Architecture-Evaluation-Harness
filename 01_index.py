#!/usr/bin/env python3
"""Stage 1 — build and persist both retrieval indexes. Nothing downstream.

Builds, once, the two things every later stage loads instead of recomputing:

    dense  : bge-base-en-v1.5 (fp16, SDPA) -> FAISS IndexFlatIP  -> artifacts/index/faiss.index
    sparse : rank_bm25 BM25Okapi over raw text -> pickle          -> artifacts/index/bm25.pkl

plus the two files that make those usable and auditable:

    artifacts/index/docmap.json      FAISS row ordinal <-> docid <-> raw text
    artifacts/index/index_meta.json  every decision + timing + version

Why IndexFlatIP and not an ANN index: at SciFact's ~5K docs, exact exhaustive
search is milliseconds AND exact. An approximate index (HNSW/IVF) would drop true
neighbours, which would show up as reduced recall — contaminating the very metric
this project measures. We could not then separate "dense retrieval underperformed"
from "the index lost the answer". Exactness here is not a performance choice, it
is a measurement-validity choice.

Usage:
    python 01_index.py                       # skips if artifacts already exist
    python 01_index.py --force               # rebuild from scratch
    python 01_index.py --chunk-policy chunked   # override the auto decision
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src.common import (  # noqa: E402
    ATTN_IMPLEMENTATION,
    BM25_PATH,
    BM25_TOKENIZER_NOTE,
    DEFAULT_BATCH_SIZE,
    DOCID_ORDER_NOTE,
    DOCMAP_PATH,
    EMBED_DIM,
    EMBED_DTYPE,
    EMBED_MODEL,
    EXPECTED_N_DOCS,
    EXPECTED_N_TEST_QUERIES,
    FAISS_PATH,
    INDEX_DIR,
    MAX_SEQ_LENGTH,
    META_PATH,
    NORMALIZATION_NOTE,
    SCIFACT_SPLIT,
    SCIFACT_URL,
    STAGE1_ARTIFACTS,
    TEXT_FIELD_NOTE,
    TRUNCATION_TOLERANCE,
    bm25_tokenize,
    doc_text,
    embed_passages,
    load_embedder,
    load_meta,
    load_scifact,
    package_versions,
    sorted_doc_ids,
    stage1_artifacts_present,
)

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


# ---------------------------------------------------------------------------
# Token-length audit -> chunk policy decision
# ---------------------------------------------------------------------------
def audit_token_lengths(texts: list[str]) -> tuple[list[int], dict]:
    """Token length of every doc under bge-base's own tokenizer, with specials."""
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)

    # The tokenizer warns loudly when a sequence exceeds model_max_length. That is
    # exactly what we are measuring, so silence it rather than have it look like a bug.
    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    enc = tok(texts, add_special_tokens=True, truncation=False, padding=False)
    hf_logging.set_verbosity(prev)

    lengths = [len(ids) for ids in enc["input_ids"]]
    arr = np.asarray(lengths)
    stats = {
        "min": int(arr.min()),
        "mean": round(float(arr.mean()), 1),
        "median": int(np.median(arr)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "n_over_limit": int((arr > MAX_SEQ_LENGTH).sum()),
        "limit": MAX_SEQ_LENGTH,
        "n_total": len(lengths),
    }
    stats["frac_over_limit"] = round(stats["n_over_limit"] / len(lengths), 5)
    return lengths, stats


def decide_chunk_policy(requested: str, stats: dict) -> tuple[str, str]:
    """Return (policy, one-line reason). `auto` prefers one vector per doc."""
    frac = stats["frac_over_limit"]
    n_over = stats["n_over_limit"]

    if requested != "auto":
        return requested, f"forced by --chunk-policy {requested}"

    if n_over == 0:
        return (
            "one_vector_per_doc",
            f"every doc fits in {MAX_SEQ_LENGTH} tokens (max observed {stats['max']}); "
            "no truncation, no chunking, no dedup",
        )
    if frac <= TRUNCATION_TOLERANCE:
        return (
            "one_vector_per_doc",
            f"{n_over}/{stats['n_total']} docs ({frac:.3%}) exceed {MAX_SEQ_LENGTH} tokens, "
            f"within the {TRUNCATION_TOLERANCE:.0%} tolerance -> truncate those, keep one vector "
            "per doc so no later stage needs chunk->doc dedup",
        )
    return (
        "chunked",
        f"{n_over} docs ({frac:.2%}) exceed {MAX_SEQ_LENGTH} tokens, above the "
        f"{TRUNCATION_TOLERANCE:.0%} tolerance -> chunk, and Stage 2 MUST collapse "
        "chunks to doc level before computing metrics against doc-level qrels",
    )


def chunk_text_by_tokens(tok, text: str, max_len: int, overlap: int) -> list[str]:
    """Fixed token-window chunker with overlap, using the embedder's own tokenizer.

    Deliberately NOT repo A's chunk_text() (reference/hybrid-rag/src/ingest.py):
    that one splits on blank-line paragraphs then sentences and counts with
    tiktoken's cl100k_base. SciFact docs are a title plus a single unbroken
    abstract — there are no paragraphs to split on, and cl100k token counts are
    not what bge's BERT tokenizer will actually do. Windowing on the model's own
    tokenizer is the only way to guarantee no chunk overflows the encoder.
    """
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    budget = max_len - 2  # leave room for [CLS] and [SEP]
    if len(ids) <= budget:
        return [text]

    step = max(1, budget - overlap)
    out: list[str] = []
    for start in range(0, len(ids), step):
        piece = ids[start : start + budget]
        if not piece:
            break
        out.append(tok.decode(piece, skip_special_tokens=True))
        if start + budget >= len(ids):
            break
    return out


def build_units(
    doc_ids: list[str],
    texts: list[str],
    lengths: list[int],
    policy: str,
    chunk_overlap: int,
) -> tuple[list[str], list[str], dict[str, list[int]], list[str]]:
    """Return (unit_texts, ordinal_to_docid, docid_to_ordinals, truncated_docids)."""
    if policy == "one_vector_per_doc":
        # Over-long docs are truncated by the encoder at max_seq_length. Record
        # exactly which ones, so Stage 2 can check whether any truncated doc is a
        # gold doc — if none are, truncation provably cannot move any metric.
        truncated = [d for d, n in zip(doc_ids, lengths) if n > MAX_SEQ_LENGTH]
        return (
            list(texts),
            list(doc_ids),
            {d: [i] for i, d in enumerate(doc_ids)},
            truncated,
        )

    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    # We tokenize without truncation on purpose (that is how we find the window
    # boundaries), so silence the "sequence longer than maximum" warning it emits —
    # otherwise correct behaviour looks like a defect in the log.
    prev_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()

    unit_texts: list[str] = []
    ordinal_to_docid: list[str] = []
    docid_to_ordinals: dict[str, list[int]] = {}

    for docid, text in zip(doc_ids, texts):
        pieces = chunk_text_by_tokens(tok, text, MAX_SEQ_LENGTH, chunk_overlap)
        ordinals = []
        for piece in pieces:
            ordinals.append(len(unit_texts))
            unit_texts.append(piece)
            ordinal_to_docid.append(docid)
        docid_to_ordinals[docid] = ordinals

    hf_logging.set_verbosity(prev_verbosity)
    return unit_texts, ordinal_to_docid, docid_to_ordinals, []


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument(
        "--chunk-policy",
        choices=("auto", "one_vector_per_doc", "chunked"),
        default="auto",
        help="auto decides from the measured token-length distribution",
    )
    ap.add_argument("--chunk-overlap", type=int, default=64, help="token overlap, chunked policy only")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()

    import faiss
    import torch

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # --- idempotency ------------------------------------------------------
    if stage1_artifacts_present() and not args.force:
        hdr("Stage 1 — SKIPPED (artifacts already present)")
        for p in STAGE1_ARTIFACTS:
            print(f"  {p.relative_to(INDEX_DIR.parent.parent)}  ({p.stat().st_size / 1024:.0f} KiB)")
        meta = load_meta()
        print(f"\n  built at        : {meta.get('created_utc')}")
        print(f"  embedder        : {meta.get('embedder', {}).get('model')}")
        print(f"  chunk_policy    : {meta.get('chunk_policy')}   dedup_needed={meta.get('dedup_needed')}")
        print(f"  faiss ntotal    : {meta.get('faiss', {}).get('ntotal')}  dim={meta.get('faiss', {}).get('dim')}")
        print("\n  Pass --force to rebuild. Nothing was changed.")
        return 0

    t_start = time.perf_counter()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- CHECK 1: corpus counts -------------------------------------------
    hdr("[1] Load SciFact (BEIR) and sanity-check the counts")
    t0 = time.perf_counter()
    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    t_load = time.perf_counter() - t0

    doc_ids = sorted_doc_ids(corpus)
    texts = [doc_text(corpus[d]) for d in doc_ids]
    n_docs = len(doc_ids)
    n_queries = len(queries)
    n_qrels_pairs = sum(len(v) for v in qrels.values())
    gold_docids = {d for q in qrels for d, rel in qrels[q].items() if rel > 0}

    print(f"  corpus docs         : {n_docs}      (published: {EXPECTED_N_DOCS})")
    print(f"  {SCIFACT_SPLIT} queries        : {n_queries}       (published: {EXPECTED_N_TEST_QUERIES})")
    print(f"  qrels (q,doc) pairs : {n_qrels_pairs}")
    print(f"  distinct gold docs  : {len(gold_docids)}")
    print(f"  queries w/ >1 gold  : {sum(1 for q in qrels if len(qrels[q]) > 1)}")
    print(f"  text field          : {TEXT_FIELD_NOTE}")
    print(f"  docid order         : {DOCID_ORDER_NOTE}")
    print(f"  load time           : {t_load:.1f}s")

    count_ok = n_docs == EXPECTED_N_DOCS and n_queries == EXPECTED_N_TEST_QUERIES
    if count_ok:
        print("  -> counts match published BEIR SciFact stats")
    else:
        print("  -> !! COUNTS DIFFER from published stats. Do not proceed until explained:")
        print(f"        got {n_docs} docs / {n_queries} queries, "
              f"expected {EXPECTED_N_DOCS} / {EXPECTED_N_TEST_QUERIES}")

    # --- token audit + policy --------------------------------------------
    hdr("[2] Token-length audit -> chunk policy")
    t0 = time.perf_counter()
    lengths, len_stats = audit_token_lengths(texts)
    t_audit = time.perf_counter() - t0
    print(f"  tokenizer           : {EMBED_MODEL} (with special tokens)")
    print(f"  encoder limit       : {MAX_SEQ_LENGTH} tokens")
    print(f"  min / median / mean : {len_stats['min']} / {len_stats['median']} / {len_stats['mean']}")
    print(f"  p95 / p99 / max     : {len_stats['p95']} / {len_stats['p99']} / {len_stats['max']}")
    print(f"  docs over limit     : {len_stats['n_over_limit']} ({len_stats['frac_over_limit']:.3%})")
    print(f"  audit time          : {t_audit:.1f}s")

    policy, reason = decide_chunk_policy(args.chunk_policy, len_stats)
    dedup_needed = policy == "chunked"
    print(f"\n  chunk_policy        : {policy}")
    print(f"  dedup_needed        : {dedup_needed}")
    print(f"  reason              : {reason}")

    unit_texts, ordinal_to_docid, docid_to_ordinals, truncated = build_units(
        doc_ids, texts, lengths, policy, args.chunk_overlap
    )
    n_units = len(unit_texts)
    print(f"  embedding units     : {n_units}" + ("" if policy == "chunked" else "  (== doc count)"))

    # Over-limit audit — runs under BOTH policies, because the number that matters
    # is "how much evidence would truncation have destroyed", and that question is
    # what justifies paying the chunking complexity (or not).
    over_limit_docids = [d for d, n in zip(doc_ids, lengths) if n > MAX_SEQ_LENGTH]
    over_limit_gold = sorted(set(over_limit_docids) & gold_docids)
    gold_pairs_at_risk = sum(
        1 for q in qrels for d, rel in qrels[q].items() if rel > 0 and d in set(over_limit_gold)
    )
    print(f"\n  docs over 512       : {len(over_limit_docids)}")
    print(f"  ...of which GOLD    : {len(over_limit_gold)} of {len(gold_docids)} gold docs "
          f"({len(over_limit_gold) / max(len(gold_docids), 1):.1%})")
    print(f"  qrels pairs at risk : {gold_pairs_at_risk} of {n_qrels_pairs}")
    if policy == "one_vector_per_doc":
        if not over_limit_gold:
            print("  -> no truncated doc is a gold doc, so truncation cannot move any metric")
        else:
            print("  -> !! gold docs ARE truncated; this is a stated limitation of the dense arm")
    else:
        print("  -> chunked instead of truncated, so no gold evidence text is discarded")

    truncation_stats = {
        "policy_truncates": policy == "one_vector_per_doc",
        "tolerance": TRUNCATION_TOLERANCE,
        "n_over_limit": len(over_limit_docids),
        "n_over_limit_gold": len(over_limit_gold),
        "over_limit_gold_docids": over_limit_gold,
        "qrels_pairs_at_risk_if_truncated": gold_pairs_at_risk,
        "n_truncated": len(truncated),
        "truncated_docids": truncated,
        "note": "truncation would affect the dense arm only; BM25 always sees full text",
    }

    chunk_stats = None
    if dedup_needed:
        counts = [len(v) for v in docid_to_ordinals.values()]
        n_multi = sum(1 for c in counts if c > 1)
        multi = [c for c in counts if c > 1]
        chunk_stats = {
            "n_docs_split": n_multi,
            "n_chunks_from_split_docs": sum(multi),
            "chunks_per_split_doc_mean": round(sum(multi) / max(len(multi), 1), 2),
            "chunks_per_split_doc_max": max(counts),
            "overlap_tokens": args.chunk_overlap,
            "window_tokens": MAX_SEQ_LENGTH - 2,
        }
        print(f"\n  docs actually split : {n_multi}")
        print(f"  chunks from those   : {sum(multi)}  "
              f"(mean {chunk_stats['chunks_per_split_doc_mean']}, max {max(counts)} per doc)")
        print("  -> !! Stage 2 MUST collapse chunks to doc level BEFORE metrics, or recall inflates")

    # --- dense embed ------------------------------------------------------
    hdr("[3] Embed passages (dense arm)")
    print(f"  model               : {EMBED_MODEL}")
    print(f"  device / dtype      : {device} / {EMBED_DTYPE}")
    print(f"  attn implementation : {ATTN_IMPLEMENTATION}   (never flash_attention_2 on sm_75)")
    print(f"  batch size          : {args.batch_size}")
    print(f"  passage convention  : raw text, NO instruction/prefix (bge, not e5)")

    t0 = time.perf_counter()
    model = load_embedder(device=device)
    t_model_load = time.perf_counter() - t0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    vectors = embed_passages(model, unit_texts, batch_size=args.batch_size)
    t_embed = time.perf_counter() - t0

    peak_alloc = peak_reserved = None
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    print(f"\n  model load          : {t_model_load:.1f}s")
    print(f"  embed wall-clock    : {t_embed:.1f}s  ({n_units / max(t_embed, 1e-9):.0f} units/s)")
    print(f"  vectors             : shape={vectors.shape} dtype={vectors.dtype}")
    print(f"  normalization       : {NORMALIZATION_NOTE}")
    norms = np.linalg.norm(vectors, axis=1)
    print(f"  L2 norm min/max     : {norms.min():.6f} / {norms.max():.6f}  (expect ~1.0)")

    if vectors.shape[1] != EMBED_DIM:
        print(f"\n  !! embedding dim {vectors.shape[1]} != expected {EMBED_DIM} — aborting")
        return 1
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("\n  !! vectors are not unit-norm; inner product would NOT equal cosine — aborting")
        return 1

    # --- CHECK 6: VRAM ----------------------------------------------------
    if device == "cuda":
        print(f"\n  peak VRAM allocated : {peak_alloc:.2f} GiB")
        print(f"  peak VRAM reserved  : {peak_reserved:.2f} GiB   (ceiling 14.5 GiB)")
        print("  -> " + ("well under the ceiling" if peak_reserved < 14.5 else "!! AT/OVER THE CEILING"))
        del model
        torch.cuda.empty_cache()

    # --- FAISS ------------------------------------------------------------
    hdr("[4] Build + persist FAISS IndexFlatIP")
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_PATH))
    t_faiss = time.perf_counter() - t0

    print(f"  index type          : IndexFlatIP (exact, exhaustive; metric=inner product)")
    print(f"  ntotal              : {index.ntotal}")
    print(f"  dim (index.d)       : {index.d}")
    print(f"  build+write         : {t_faiss:.2f}s")
    print(f"  file                : {FAISS_PATH}  ({FAISS_PATH.stat().st_size / 2**20:.1f} MiB)")

    # CHECK 2
    if index.ntotal != n_units:
        print(f"  !! ntotal {index.ntotal} != embedded units {n_units} — aborting")
        return 1
    print(f"  -> ntotal == embedded units ({n_units}) OK")
    if policy == "one_vector_per_doc":
        if index.ntotal != n_docs:
            print(f"  !! ntotal {index.ntotal} != doc count {n_docs} under one_vector_per_doc — aborting")
            return 1
        print(f"  -> ntotal == doc count ({n_docs}) OK")
    if index.d != EMBED_DIM:
        print(f"  !! index.d {index.d} != {EMBED_DIM} — aborting")
        return 1

    # --- BM25 -------------------------------------------------------------
    hdr("[5] Build + persist BM25 (sparse arm)")
    from rank_bm25 import BM25Okapi

    t0 = time.perf_counter()
    # BM25 is always DOC-LEVEL and never chunked — it has no input length limit,
    # so it sees the full text even for docs the dense arm had to truncate. That
    # asymmetry is real and is recorded in index_meta.json.
    tokenized = [bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    bundle = {
        "schema_version": 1,
        "bm25": bm25,
        "doc_ids": doc_ids,  # score-array ordering: get_scores()[j] -> doc_ids[j]
        "tokenizer_note": BM25_TOKENIZER_NOTE,
        "text_field_note": TEXT_FIELD_NOTE,
        "docid_order_note": DOCID_ORDER_NOTE,
        "rank_bm25_version": package_versions()["rank-bm25"],
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_bm25 = time.perf_counter() - t0

    avg_tokens = sum(len(t) for t in tokenized) / len(tokenized)
    vocab = len({tok for t in tokenized for tok in t})
    print(f"  implementation      : rank_bm25.BM25Okapi (k1/b library defaults)")
    print(f"  tokenizer           : {BM25_TOKENIZER_NOTE}")
    print(f"  granularity         : doc-level, unchunked, untruncated (full text)")
    print(f"  docs                : {len(tokenized)}")
    print(f"  avg tokens/doc      : {avg_tokens:.1f}")
    print(f"  vocabulary size     : {vocab}")
    print(f"  build+pickle        : {t_bm25:.2f}s")
    print(f"  file                : {BM25_PATH}  ({BM25_PATH.stat().st_size / 2**20:.1f} MiB)")

    # --- docmap -----------------------------------------------------------
    hdr("[6] Write docmap.json (ordinal <-> docid <-> text)")
    docmap = {
        "schema_version": 1,
        "chunk_policy": policy,
        "dedup_needed": dedup_needed,
        "n_units": n_units,
        "n_docs": n_docs,
        "ordinal_to_docid": ordinal_to_docid,
        "docid_to_ordinals": docid_to_ordinals,
        "docid_to_text": {d: t for d, t in zip(doc_ids, texts)},
        # The exact string embedded at each ordinal. Under `chunked` this is the
        # decoded chunk, which differs from the doc text — and without it the
        # alignment proof cannot run, which is the whole reason it is persisted.
        "ordinal_to_text": unit_texts,
    }
    with open(DOCMAP_PATH, "w") as f:
        json.dump(docmap, f)
    print(f"  ordinals            : {len(ordinal_to_docid)}")
    print(f"  docids              : {len(docid_to_ordinals)}")
    print(f"  file                : {DOCMAP_PATH}  ({DOCMAP_PATH.stat().st_size / 2**20:.1f} MiB)")

    # in-process alignment assertion; verify_stage1.py proves it again from disk
    for ordinal, did in enumerate(ordinal_to_docid):
        if ordinal not in docid_to_ordinals[did]:
            print(f"  !! alignment broken at ordinal {ordinal} (docid {did}) — aborting")
            return 1
    print("  -> every ordinal round-trips through docid_to_ordinals OK")

    # --- meta -------------------------------------------------------------
    t_total = time.perf_counter() - t_start
    meta = {
        "schema_version": 1,
        "stage": "01_index",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": {
            "name": "scifact",
            "source": "beir",
            "url": SCIFACT_URL,
            "split": SCIFACT_SPLIT,
            "n_docs": n_docs,
            "n_queries": n_queries,
            "n_qrels_pairs": n_qrels_pairs,
            "n_gold_docs": len(gold_docids),
            "expected_n_docs": EXPECTED_N_DOCS,
            "expected_n_queries": EXPECTED_N_TEST_QUERIES,
            "counts_match_published": count_ok,
            "text_field": TEXT_FIELD_NOTE,
            "docid_order": DOCID_ORDER_NOTE,
        },
        "embedder": {
            "model": EMBED_MODEL,
            "dim": int(vectors.shape[1]),
            "dtype": EMBED_DTYPE,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "max_seq_length": MAX_SEQ_LENGTH,
            "device": device,
            "batch_size": args.batch_size,
            "passage_convention": "raw text, no instruction (bge convention, not e5 prefixes)",
            "query_convention": "bge retrieval instruction prepended at search time (Stage 2)",
        },
        "chunk_policy": policy,
        "dedup_needed": dedup_needed,
        "chunk_policy_reason": reason,
        "chunk_overlap_tokens": args.chunk_overlap if dedup_needed else None,
        "n_units": n_units,
        "token_lengths": len_stats,
        "truncation": truncation_stats,
        "chunking": chunk_stats,
        "normalization": True,
        "normalization_note": NORMALIZATION_NOTE,
        "faiss": {
            "index_type": "IndexFlatIP",
            "exact": True,
            "metric": "inner_product (== cosine, vectors are unit-norm)",
            "ntotal": int(index.ntotal),
            "dim": int(index.d),
            "rationale": (
                "~5K docs: exact brute force is milliseconds and exact. An ANN index would "
                "inject recall error into the metric being measured."
            ),
        },
        "bm25": {
            "implementation": "rank_bm25.BM25Okapi",
            "tokenizer": BM25_TOKENIZER_NOTE,
            "granularity": "doc-level, unchunked, untruncated",
            "n_docs": len(tokenized),
            "avg_tokens_per_doc": round(avg_tokens, 1),
            "vocab_size": vocab,
            "rank_bm25_version": bundle["rank_bm25_version"],
            "pickle_note": "pickle is tied to the rank_bm25 version above; rebuild if it changes",
        },
        "timings_sec": {
            "load_scifact": round(t_load, 2),
            "token_audit": round(t_audit, 2),
            "model_load": round(t_model_load, 2),
            "embed": round(t_embed, 2),
            "faiss_build_write": round(t_faiss, 2),
            "bm25_build_pickle": round(t_bm25, 2),
            "total": round(t_total, 2),
        },
        "peak_vram_gib": {
            "allocated": round(peak_alloc, 3) if peak_alloc is not None else None,
            "reserved": round(peak_reserved, 3) if peak_reserved is not None else None,
            "ceiling": 14.5,
        },
        "versions": package_versions(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    hdr("Stage 1 complete")
    print(f"  chunk_policy   : {policy}  (dedup_needed={dedup_needed})")
    print(f"  faiss ntotal   : {index.ntotal}   dim: {index.d}")
    print(f"  bm25 docs      : {len(tokenized)}")
    print(f"  embed time     : {t_embed:.1f}s")
    if peak_reserved is not None:
        print(f"  peak VRAM      : {peak_reserved:.2f} GiB reserved / {peak_alloc:.2f} GiB allocated")
    print(f"  TOTAL          : {t_total:.1f}s")
    print(f"\n  meta written   : {META_PATH}")
    print("\n  Next: python verify_stage1.py   (fresh-process reload + round-trip + smoke)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

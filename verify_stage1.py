#!/usr/bin/env python3
"""Stage 1 verification — runs in a FRESH process, loading only from disk.

This script deliberately shares no in-memory state with 01_index.py. That is the
point of it: every later stage will reload these artifacts cold, so the reload
path is what has to be proven, not the objects that happened to be in RAM when
they were built.

Checks:
  [3] reload FAISS + BM25 + docmap + meta from disk, cold
  [4] round-trip 3 docids: ordinal -> docid -> text, and back, in both directions
      plus the alignment PROOF: re-embed the doc and compare against the vector
      FAISS actually stored at that ordinal
  [5] smoke retrieval — one query, dense top-5 and BM25 top-5, resolved through
      docmap. Wiring only. No metrics are computed here; that is Stage 2.

Usage:
    python verify_stage1.py
    python verify_stage1.py --qid 13 --docids 4983,23389,42212
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src.common import (  # noqa: E402
    BGE_QUERY_INSTRUCTION,
    BM25_PATH,
    DOCMAP_PATH,
    EMBED_DIM,
    EXPECTED_N_DOCS,
    EXPECTED_N_TEST_QUERIES,
    FAISS_PATH,
    META_PATH,
    bm25_tokenize,
    embed_queries,
    embed_passages,
    load_bm25,
    load_docmap,
    load_faiss,
    load_meta,
    load_embedder,
    load_scifact,
    stage1_artifacts_present,
)

BANNER = "=" * 74
ALIGNMENT_COS_MIN = 0.99  # fp16 encode is not bit-exact across batch shapes


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def snippet(text: str, n: int = 88) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qid", default=None, help="SciFact query id for the smoke check")
    ap.add_argument("--docids", default=None, help="comma-separated docids for the round-trip check")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    if not stage1_artifacts_present():
        print("Stage 1 artifacts missing. Run: python 01_index.py")
        return 1

    failures: list[str] = []

    # ----------------------------------------------------------------- [3]
    hdr("[3] Cold reload from disk (fresh process)")
    try:
        index = load_faiss()
        print(f"  faiss.read_index    : OK  ntotal={index.ntotal}  dim={index.d}  "
              f"type={type(index).__name__}")
    except Exception as exc:
        print(f"  faiss.read_index    : FAILED — {type(exc).__name__}: {exc}")
        return 1

    try:
        bundle = load_bm25()
        bm25 = bundle["bm25"]
        bm25_doc_ids = bundle["doc_ids"]
        print(f"  bm25 unpickle       : OK  docs={len(bm25_doc_ids)}  "
              f"rank_bm25={bundle.get('rank_bm25_version')}")
    except Exception as exc:
        print(f"  bm25 unpickle       : FAILED — {type(exc).__name__}: {exc}")
        return 1

    try:
        docmap = load_docmap()
        print(f"  docmap.json         : OK  n_units={docmap.n_units}  "
              f"policy={docmap.chunk_policy}  dedup_needed={docmap.dedup_needed}")
    except Exception as exc:
        print(f"  docmap.json         : FAILED — {type(exc).__name__}: {exc}")
        return 1

    meta = load_meta()
    print(f"  index_meta.json     : OK  built {meta['created_utc']}")
    for p in (FAISS_PATH, BM25_PATH, DOCMAP_PATH, META_PATH):
        print(f"      {p.name:<18} {p.stat().st_size / 2**20:7.2f} MiB")

    # ----------------------------------------------------- [1] recap + [2]
    hdr("[1] Corpus counts (re-read from the cached BEIR download)")
    corpus, queries, qrels = load_scifact()
    n_docs, n_queries = len(corpus), len(queries)
    n_pairs = sum(len(v) for v in qrels.values())
    print(f"  corpus docs         : {n_docs}  (published {EXPECTED_N_DOCS})")
    print(f"  test queries        : {n_queries}   (published {EXPECTED_N_TEST_QUERIES})")
    print(f"  qrels pairs         : {n_pairs}")
    if n_docs != EXPECTED_N_DOCS or n_queries != EXPECTED_N_TEST_QUERIES:
        failures.append("corpus counts differ from published SciFact stats")
        print("  -> !! MISMATCH vs published stats")
    else:
        print("  -> matches published stats")

    hdr("[2] FAISS shape assertions")
    print(f"  index.ntotal        : {index.ntotal}")
    print(f"  index.d             : {index.d}  (expect {EMBED_DIM} for bge-base)")
    print(f"  docmap n_units      : {docmap.n_units}")
    print(f"  meta n_units        : {meta['n_units']}")
    checks = [
        ("ntotal == docmap n_units", index.ntotal == docmap.n_units),
        ("ntotal == meta n_units", index.ntotal == meta["n_units"]),
        ("ntotal == len(ordinal_to_docid)", index.ntotal == len(docmap.ordinal_to_docid)),
        (f"index.d == {EMBED_DIM}", index.d == EMBED_DIM),
        ("bm25 docs == corpus docs", len(bm25_doc_ids) == n_docs),
    ]
    if docmap.chunk_policy == "one_vector_per_doc":
        checks.append(("ntotal == corpus docs (one_vector_per_doc)", index.ntotal == n_docs))
    for label, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(label)

    # ----------------------------------------------------------------- [4]
    hdr("[4] Round-trip: ordinal <-> docid <-> text")
    if args.docids:
        probe_docids = [d.strip() for d in args.docids.split(",")]
    else:
        ords = [0, docmap.n_units // 2, docmap.n_units - 1]
        probe_docids = [docmap.docid(o) for o in ords]
        # Position-chosen probes are all likely to be single-chunk docs, which cannot
        # exercise the failure mode chunking actually introduces: a doc's 2nd or 3rd
        # vector landing at the wrong ordinal. Force multi-chunk docs into the probe
        # set — the deepest one (most chunks) and a shallow one (fewest, i.e. 2).
        if docmap.dedup_needed:
            multi = sorted(
                (d for d, o in docmap.docid_to_ordinals.items() if len(o) > 1),
                key=lambda d: (-len(docmap.ordinals(d)), d),
            )
            for extra in ([multi[0], multi[-1]] if multi else []):
                if extra not in probe_docids:
                    probe_docids.append(extra)
            print(f"  multi-chunk docs    : {len(multi)}; probing the deepest "
                  f"({len(docmap.ordinals(multi[0])) if multi else 0} chunks) and a 2-chunk doc")

    print(f"  probing docids      : {probe_docids}\n")
    for docid in probe_docids:
        if docid not in docmap.docid_to_ordinals:
            print(f"  FAIL  docid {docid} not in docmap")
            failures.append(f"docid {docid} missing from docmap")
            continue

        ordinals = docmap.ordinals(docid)
        back = [docmap.docid(o) for o in ordinals]
        fwd_ok = all(b == docid for b in back)
        in_corpus = docid in corpus
        text = docmap.text(docid)
        bm25_pos = bm25_doc_ids.index(docid) if docid in bm25_doc_ids else -1

        print(f"  docid {docid}  ({len(ordinals)} chunk{'s' if len(ordinals) != 1 else ''})")
        print(f"      ordinals            : {ordinals}")
        print(f"      ordinal->docid back : {back}   {'OK' if fwd_ok else 'FAIL'}")
        if len(ordinals) > 1:
            # build_units() appends a doc's chunks consecutively, so a gap here means
            # the ordinal space and the chunk order have come apart.
            contiguous = ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))
            print(f"      ordinals contiguous : {contiguous}   {'OK' if contiguous else 'FAIL'}")
            if not contiguous:
                failures.append(f"docid {docid} chunk ordinals are not contiguous: {ordinals}")
        print(f"      bm25 row            : {bm25_pos}")
        print(f"      in BEIR corpus      : {in_corpus}")
        print(f"      text                : {snippet(text)}")
        if not fwd_ok:
            failures.append(f"docid {docid} ordinal round-trip broken")
        if not in_corpus:
            failures.append(f"docid {docid} not in BEIR corpus")
        if bm25_pos < 0:
            failures.append(f"docid {docid} missing from bm25 doc_ids")
        print()

    # The real alignment proof: does the vector FAISS stored at this ordinal
    # actually encode THIS document? A shuffled docmap passes every check above
    # and fails only here — which is exactly the silent corruption we fear.
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_embedder(device=device)
    print("  Alignment proof — re-embed the exact unit text and compare to the stored vector:")
    if docmap.ordinal_to_text is None:
        print("      CANNOT RUN — this docmap.json predates ordinal_to_text.")
        print("      Rebuild with `python 01_index.py --force` to enable the proof.")
        failures.append("alignment proof unavailable: docmap.json has no ordinal_to_text")
    else:
        # Probe EVERY ordinal of each docid, not just the first — under chunking a
        # doc owns several rows and any one of them could be misplaced.
        probe_ordinals: list[tuple[int, str]] = []
        for docid in probe_docids:
            for o in docmap.ordinals(docid):
                probe_ordinals.append((o, docid))

        for ordinal, docid in probe_ordinals:
            stored = index.reconstruct(ordinal)
            fresh = embed_passages(model, [docmap.unit_text(ordinal)], progress=False)[0]
            cos = float(np.dot(stored, fresh))
            ok = cos >= ALIGNMENT_COS_MIN
            print(f"      ordinal {ordinal:<5} docid {docid:<10} cos(stored, re-embedded) = {cos:.6f}  "
                  f"{'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"ordinal {ordinal}/docid {docid} vector does not match its unit text")

        # Negative control: the first probe's stored vector against a DIFFERENT
        # unit's text must NOT score ~1.0. Without this, a comparison bug that
        # always returned 1.0 would read as a clean pass.
        if len(probe_ordinals) > 1:
            o0, d0 = probe_ordinals[0]
            o1, d1 = probe_ordinals[-1]
            v0 = index.reconstruct(o0)
            v1 = embed_passages(model, [docmap.unit_text(o1)], progress=False)[0]
            control = float(np.dot(v0, v1))
            print(f"      negative control  cos(vec[ord {o0}, doc {d0}], text[ord {o1}, doc {d1}]) "
                  f"= {control:.6f}  (must be clearly < 1.0)")
            if control >= ALIGNMENT_COS_MIN:
                failures.append("negative control scored ~1.0 — the comparison itself is broken")

    # ----------------------------------------------------------------- [5]
    hdr("[5] Smoke retrieval — ONE query, wiring check only, NO metrics")
    qids = sorted(queries.keys(), key=lambda q: int(q) if q.isdigit() else q)
    qid = args.qid or next((q for q in qids if qrels.get(q)), qids[0])
    if qid not in queries:
        print(f"  qid {qid} not in the {len(queries)} test queries")
        return 1
    qtext = queries[qid]
    gold = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}

    print(f"  qid                 : {qid}")
    print(f"  query               : {qtext}")
    print(f"  gold docids (qrels) : {sorted(gold)}")
    print(f"  query instruction   : {BGE_QUERY_INSTRUCTION!r}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    qvec = embed_queries(model, [qtext], use_instruction=True)
    print(f"  query vector        : shape={qvec.shape} dtype={qvec.dtype} "
          f"norm={float(np.linalg.norm(qvec)):.6f}")

    scores, ordinals = index.search(qvec, args.top_k)
    print(f"\n  DENSE top-{args.top_k} (FAISS IndexFlatIP, score == cosine):")
    for rank, (o, s) in enumerate(zip(ordinals[0], scores[0]), start=1):
        did = docmap.docid(int(o))
        resolves = did in docmap.docid_to_text and did in corpus
        flag = " <-- GOLD" if did in gold else ""
        print(f"    {rank}. ordinal={int(o):<5} docid={did:<8} cos={float(s):.4f} "
              f"resolves={resolves}{flag}")
        print(f"       {snippet(docmap.text(did), 76)}")
        if not resolves:
            failures.append(f"dense hit docid {did} does not resolve through docmap")

    bm25_scores = bm25.get_scores(bm25_tokenize(qtext))
    top = np.argsort(-bm25_scores)[: args.top_k]
    print(f"\n  BM25 top-{args.top_k} (rank_bm25 BM25Okapi, raw untruncated text):")
    for rank, j in enumerate(top, start=1):
        did = bm25_doc_ids[int(j)]
        resolves = did in docmap.docid_to_text and did in corpus
        flag = " <-- GOLD" if did in gold else ""
        print(f"    {rank}. row={int(j):<5} docid={did:<8} bm25={float(bm25_scores[j]):.4f} "
              f"resolves={resolves}{flag}")
        print(f"       {snippet(docmap.text(did), 76)}")
        if not resolves:
            failures.append(f"bm25 hit docid {did} does not resolve through docmap")

    dense_ids = [docmap.docid(int(o)) for o in ordinals[0]]
    sparse_ids = [bm25_doc_ids[int(j)] for j in top]
    print(f"\n  overlap between the two top-{args.top_k} lists: "
          f"{len(set(dense_ids) & set(sparse_ids))}/{args.top_k}  "
          "(an observation, NOT a metric — Stage 2 owns all measurement)")

    # ----------------------------------------------------------------- [6]
    hdr("[6] Resource recap")
    pv = meta.get("peak_vram_gib", {})
    tm = meta.get("timings_sec", {})
    print(f"  Stage 1 embed peak  : {pv.get('reserved')} GiB reserved / "
          f"{pv.get('allocated')} GiB allocated  (ceiling {pv.get('ceiling')})")
    print(f"  Stage 1 embed time  : {tm.get('embed')}s   total {tm.get('total')}s")
    if device == "cuda":
        print(f"  this script's peak  : {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved")

    hdr("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All Stage 1 checks passed. Indexes are persisted, aligned, and reloadable.")
    print("  Stage 2 (retrieval + metrics) is unblocked — but do not start it until asked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

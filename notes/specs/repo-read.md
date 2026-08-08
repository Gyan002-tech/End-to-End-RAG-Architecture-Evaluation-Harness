# Reference-repo read — Stage 0

**Status:** study-only pass. Neither repo was run to completion; no number from either
repo enters our tables. Read against `slotA-methodology.md` §6 and §9 Step 0.

| | Repo | Commit read | Role |
|---|---|---|---|
| **A** | `tim-ponomarev/hybrid-rag` | `78ab157` *chore: CPU smoke test + default config* | pipeline skeleton + judge rubric |
| **B** | `slavadubrov/rag-evals-demo` | `5709a4b` *docs* | SciFact metric wiring + pinned known-good values |

Both are vendored under `reference/` (gitignored). Repo A is 0 stars / v0.1.0 / single
author — **trust the structure, verify the logic.** Two logic defects found; see §5.

---

## 1. Repo A — what each module does

Files actually present in `reference/hybrid-rag/src/`: `ingest.py`, `retrieval.py`,
`retrieval_inmemory.py`, `rerank.py`, `generate.py`, `judge.py`, `pipeline.py`, `eval.py`.
(`retrieval_inmemory.py` and `pipeline.py` are not in the methodology's reuse map but are
load-bearing — the first is what makes the smoke test run without network, the second is
where per-stage latency timing already exists.)

| Module | What it does |
|---|---|
| `ingest.py` | Recursive paragraph→sentence chunker (512 tok, 20% overlap, `tiktoken` cl100k_base) → embeds with `e5-large-v2` in batches of 64 → upserts to a Qdrant collection, and writes a parallel `corpus.jsonl` for the BM25 lane. |
| `retrieval.py` | Three retrievers + the RRF function. `BM25Retriever` (in-memory `BM25Okapi`), `DenseRetriever` (Qdrant + `e5-large-v2`, with `query:`/`passage:` prefixes), `HybridRetriever` (fuses the two lanes' rank lists). Heavy deps are import-guarded, so BM25 + RRF work with only numpy + rank_bm25. |
| `retrieval_inmemory.py` | `InMemoryDenseRetriever`: numpy cosine over a dense matrix, plus `_HashEmbedder` — a deterministic md5-bit-projection stub embedder that is **not semantically meaningful**. Exists purely so the smoke test needs no model download. |
| `rerank.py` | One `CrossEncoderReranker` wrapping `sentence_transformers.CrossEncoder` on `ms-marco-MiniLM-L-6-v2`; scores all (query, doc) pairs in batches of 32, re-sorts, returns top-k. No model swapping, no unload path. |
| `generate.py` | Formats retrieved docs as a numbered `[1] (doc_id=…)` context block and calls OpenAI chat completions (`gpt-4o-mini`, temp 0.2) under a strict grounding system prompt ("cite inline, say I don't know, never use training knowledge"). |
| `judge.py` | LLM-as-judge scoring faithfulness / relevance / coverage each 1–5 with a one-sentence rationale, via `gpt-4o-mini` with `response_format=json_object`, temp 0.0, primed by **5 hand-written few-shot examples** injected as alternating user/assistant turns. |
| `pipeline.py` | Wires hybrid → rerank → generate and records `retrieval_ms` / `rerank_ms` / `generation_ms` / `total_ms` via `time.perf_counter()`. |
| `eval.py` | Loops a query file, computes nDCG@10 / MRR@10 / Recall@50 plus the three judge axes, dumps per-query JSONL and prints a summary. Hand-written metric math (see §5 — one of them is wrong). |

**Import style caveat:** `src/` uses flat imports (`from retrieval import …`), not package
relative. `scripts/smoke_test.py` compensates by inserting `<repo>/src` into `sys.path`.
Anything we lift out of `src/` must either keep that path hack or have its imports rewritten.

---

## 2. The exact RRF implementation in repo A's `retrieval.py`

Verbatim, `reference/hybrid-rag/src/retrieval.py:36-55`:

```python
def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

Properties that matter, stated precisely:

- **Signature is rank lists only.** It takes `list[list[doc_id]]` — raw BM25 / cosine scores
  never enter the function. This is *why* RRF works across incomparable score scales, and is
  the §8 checklist item "I can explain why RRF uses rank not score."
- **`k = 60`**, Cormack et al.'s canonical default. Larger `k` flattens the rank discount
  (later ranks matter relatively more); smaller `k` sharpens top-rank dominance.
- **1-indexed rank** (`enumerate(..., start=1)`), so the best doc in a lane contributes
  `1/61`, not `1/60`.
- **Union, not intersection.** A doc present in one lane and absent from the other simply
  receives no second term — there is no absence penalty.
- **N lanes, unweighted.** No per-lane weighting and no division by lane count, so absolute
  fused scores are not comparable across different numbers of lanes (ordering within one call
  is unaffected).
- **Tie-break is insertion order.** Python's `sorted` is stable and `scores` is a plain dict,
  so equal-scored docs keep first-seen order — i.e. lane order in the `rankings` argument
  decides ties. Deterministic, but an artifact of argument order rather than a principled rule.
- **Output length is the union size**, unbounded; `HybridRetriever.search` is what slices to
  `top_k` (`retrieval.py:155`).

Repo B's `retrieval/hybrid_rrf.py:20-28` is the same formula with `defaultdict(float)` and
`reverse=True`. **Two independent implementations agree** — that is our cross-check that the
formula itself is right, and it is the version we re-derive from memory.

### Worked example (hand-computed from B's `tests/test_rrf.py`, k=60)

```
dense  = [d3, d7, d1, d4, d2, d9, d10]
sparse = [d2, d3, d8, d1, d11, d4, d6]

d3 : 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522   <- rank 1 dense, rank 2 sparse
d2 : 1/65 + 1/61 = 0.015385 + 0.016393 = 0.031778
d1 : 1/63 + 1/64 = 0.015873 + 0.015625 = 0.031498
d4 : 1/64 + 1/66 = 0.015625 + 0.015152 = 0.030777
d7 : 1/62                              = 0.016129   <- one lane only
```

fused top-3 = `d3, d2, d1`, and B pins exactly that plus `score(d3) == 1/61 + 1/62`.

**The interview line is in the last two rows:** `d1`, which is merely 3rd and 4th, beats `d7`,
which is 2nd in one lane — because two mediocre agreements outweigh one strong single-lane
vote. That agreement bonus *is* the fusion effect, and it is visible with no scores anywhere.

---

## 3. The three divergences we deliberately make from repo A

Each is a headline contribution, not a copy (methodology §6).

| # | Axis | Repo A does | We do | Why it is ours, and what it costs |
|---|---|---|---|---|
| 1 | **Judge** | OpenAI `gpt-4o-mini` over the API (`judge.py:83`), 5 few-shot examples, `response_format=json_object` | Local **Qwen2.5-7B-Instruct**, ~14 GB fp16, running **alone** in stage 05 (reranker + generator already unloaded); 4-bit via bitsandbytes if tight | Self-contained and defensible — no API key, no vendor drift, every number reproducible on our own hardware. **Cost:** we lose guaranteed JSON-mode, so structured output must be prompted + parsed defensively; and the judge's own reliability becomes ours to prove → the 30–50 hand-labeled set + Cohen's κ / QWK is now **part of the deliverable**, not optional. If we quantize, the bullet says so. |
| 2 | **Reranker** | one fixed `cross-encoder/ms-marco-MiniLM-L-6-v2` (2019-era, `rerank.py:18`), no unload path | **two arms**: `BAAI/bge-reranker-v2-m3` (cross-encoder) **and** `BAAI/bge-reranker-v2-gemma` (2.5 B LLM reranker, ~5 GB fp16), model-swappable, loaded one at a time and explicitly unloaded between | Turns a fixed component into a **measured axis**. LLM-reranker vs. cross-encoder at stated latency multiplier is the 2026 differentiator (§4). **Cost:** A's `rerank.py` core is rewritten, not adapted — we keep the interface, rebuild the logic, and add the staging/unload machinery the T4 forces. |
| 3 | **Dataset & shape** | 4 hand-picked LegalBench rows, one pre-chosen pipeline, metrics bolted on afterward | **SciFact (BEIR)** with ground-truth qrels, as a **3×3 retriever × reranker grid** → quality-vs-latency Pareto frontier → knee | Removes hand annotation (every recall/nDCG is real), and removes the pre-selected pipeline: the deliverable becomes *the evidence for* the choice. Also licenses the "simpler BM25 won" result as a finding rather than a failure. **Cost:** 9 cells not 1 — which is exactly why Phase 1 runs with no generator at all. |

Consequence for the reuse map: `retrieval.py` (RRF + BM25) is the only piece we take
near-whole. `ingest.py` loses Qdrant (→ FAISS `IndexFlatIP`) and its corpus loader
(→ BEIR SciFact) but keeps the chunker. `eval.py`'s metric math is **not** reusable (§5).

---

## 4. Repo B — the pinned known-good values

**File:** `reference/rag-evals-demo/tests/test_retrieval_metrics.py`
**Implementation under test:** `reference/rag-evals-demo/src/rag_evals/evaluation/retrieval.py`

Fixture (4 queries, 10-doc runs, binary relevance):

```python
GOLD = {"q1": {"d3"}, "q2": {"d7", "d2"}, "q3": {"d11"}, "q4": {"d5"}}

RUNS = {
    "q1": ["d8", "d3", "d1", "d4", "d2", "d9", "d6", "d10", "d12", "d13"],
    "q2": ["d2", "d6", "d4", "d7", "d1", "d3", "d8", "d11", "d5", "d9"],
    "q3": ["d11", "d2", "d3", "d4", "d1", "d6", "d7", "d8", "d10", "d12"],
    "q4": ["d1", "d2", "d3", "d6", "d8", "d9", "d10", "d12", "d13", "d14"],
}
```

| Metric | Pinned value | tolerance |
|---|---|---|
| **Recall@5** | **0.750** | `abs=1e-3` |
| **MRR** | **0.625** | `abs=1e-3` |
| **nDCG@5** | **0.627** | `abs=1e-3` |

A fourth test asserts all three at once via `evaluate_runs(RUNS, GOLD, k=5)`, plus `k == 5`
and `n_queries == 4`.

### Hand-verified — these reproduce, and they pin a *convention*

Worked per query (k=5), computed by hand, not by running the repo:

| q | first rel rank | Recall@5 | RR | DCG@5 | IDCG@5 | nDCG@5 |
|---|---|---|---|---|---|---|
| q1 | 2 | 1/1 = 1.0 | 0.500 | 1/log₂3 = 0.6309 | 1.0 | 0.6309 |
| q2 | 1 (d2), also d7@4 | 2/2 = 1.0 | 1.000 | 1 + 1/log₂5 = 1.4307 | 1 + 1/log₂3 = 1.6309 | 0.8772 |
| q3 | 1 | 1/1 = 1.0 | 1.000 | 1.0 | 1.0 | 1.0000 |
| q4 | — (d5 absent from the run entirely) | 0.0 | 0.000 | 0 | 1.0 | 0.0000 |
| | **mean** | **0.750** ✓ | **0.625** ✓ | | | **0.627** ✓ |

The nDCG figure only lands on 0.627 under B's IDCG convention
(`retrieval.py:62-63`): **IDCG is built from `min(k, |gold|)` ideal hits**, independent of what
was actually retrieved. q2 needs `IDCG = 1 + 1/log₂3`, not `1`. Using repo A's convention
instead gives a different number — see §5.

Two more conventions the fixture bakes in, both of which we must match or consciously depart from:

- **MRR is untruncated** — B's `reciprocal_rank` scans the whole run, with no `@k` cutoff
  (A's `mrr_at_k` truncates). This fixture cannot tell them apart (q1's hit is at rank 2, and
  q4 has no hit anywhere), so 0.625 verifies both. We must still pick one and say which.
- **Chunk→doc dedup happens before scoring** — `_unique()` collapses repeated doc_ids
  (`retrieval.py:15-25`). If we chunk SciFact abstracts into more than one chunk per doc, we
  must dedup to doc level before metrics or recall inflates against doc-level qrels.

**Use:** in Stage 2 we hand-write `recall_at_k` / `reciprocal_rank` / `ndcg_at_k` from memory,
assert them against this fixture, *then* cross-check against `ranx` / `pytrec_eval`, and only
then trust a grid cell. Three implementations agreeing is the gate.

`tests/test_rrf.py` additionally pins the RRF worked example reproduced in §2 above
(top-3 `d3, d2, d1`; `score(d3) == 1/61 + 1/62`) — reuse that as our RRF unit test.

---

## 5. Two defects found in repo A (do not inherit these)

**5a. `eval.py`'s nDCG is inflated whenever recall@k < 1 and a query has >1 gold doc.**
`reference/hybrid-rag/src/eval.py:27-31` builds the ideal ranking from the *retrieved* list's
own labels:

```python
rel = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]
dcg = dcg_at_k(rel, k)
ideal = dcg_at_k(sorted(rel, reverse=True), k)   # <-- ideal from what we FOUND
```

If a query has 2 gold docs and only 1 appears in top-k, `ideal` becomes `1.0` instead of
`1 + 1/log₂3 = 1.6309`, so a nDCG of 0.613 is reported as **1.000**. On B's q2 fixture the two
conventions happen to coincide (both gold docs are inside top-5), which is exactly why this
would slip through a casual check. SciFact has multi-evidence claims, so this would bite us.
**Action:** use B's convention (`min(k, |gold|)`) and let `pytrec_eval` arbitrate.

**5b. `dcg_at_k` uses linear gain, not the exponential form.**
`eval.py:23-24` computes `Σ rel_i / log₂(i+2)`, whereas methodology §5 specifies
`Σ (2^rel_i − 1) / log₂(i+1)`. For **binary** relevance these are identical
(`2¹−1 = 1`), and both index the discount so rank 1 divides by `log₂2 = 1`, so there is no bug
today. It becomes one the moment graded relevance enters. **Action:** write the exponential
form, since that is the definition we must be able to reproduce from memory.

Lesser observations, not defects: `BM25Retriever` tokenizes with bare `.lower().split()`
(no stemming, no stopword removal, punctuation clings to tokens) — plausibly *fine* and even
favourable for SciFact's claim↔evidence term overlap, but it is a variable we own and should
state; and `search()` never filters zero-score hits, so it always returns `top_k` docs however
irrelevant.

---

## 6. Repo A's smoke test — what it is and what "correct" looks like

**File:** `reference/hybrid-rag/scripts/smoke_test.py`. Inserts `<repo>/src` on `sys.path`, so
it runs from any cwd. **Needs only `numpy` + `rank_bm25`** — `retrieval.py`'s `qdrant_client`
and `sentence_transformers` imports are `try/except`-guarded (`retrieval.py:12-24`), and the
dense lane is the hash-stub `InMemoryDenseRetriever`. No server, no API key, no download, ~5s.

It indexes a 10-doc mini corpus, runs 5 queries, and for each one prints the BM25 top-3, the
dense top-3, and the RRF top-3 (`k=60`), checking whether an expected doc_id is in the fused
top-3.

**The pass bar is deliberately low and the script says why:** the dense lane is a hash
embedder, not a real encoder, so semantic queries are *expected* to miss. The hard gate is
`if passed < total * 0.4: raise SystemExit(1)` (`smoke_test.py:106`) — i.e. **≥ 2 of 5 must
pass**; the point is that the plumbing runs, not that retrieval is good.

**What to check in the output (this is the real acceptance criterion, not the pass count):**
the `RRF top-3` line must be a *merge* of the two lanes, not a copy of either — it should
contain at least one doc_id that is not in the BM25 top-3, and its ordering must reward docs
appearing in both lists (per §2's agreement bonus). A `RRF top-3` identical to `BM25 top-3` on
every query would mean the dense ranking is not reaching the fusion step.

---

## 7. Verification status — NOT YET RUN

Per the working agreement, nothing in this repo is executed locally; Colab is the runtime.
These three gates are **open** and must be closed in Colab before Stage 1:

- [ ] `env_check.py` runs clean and reports a CUDA GPU (record the exact device name; flag it if not a T4/sm_75).
- [ ] `python reference/hybrid-rag/scripts/smoke_test.py` completes, and the RRF line satisfies §6.
- [ ] B's fixture reproduces (`0.750 / 0.625 / 0.627`) — deferred to Stage 2, when our own metric code exists.

Exact cells are in `notes/colab-stage0.md`.

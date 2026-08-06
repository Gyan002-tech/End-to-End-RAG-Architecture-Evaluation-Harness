# Stage 1 — index build: decisions and verification

Scope: build and persist the two indexes. No retrieval, no reranking, no metrics.
Everything here is derived from `slotA-methodology.md` §9 step 1 and `notes/repo-read.md`.

Artifacts produced, all under `artifacts/index/` (gitignored, back up to Drive):

| File | Contents |
|---|---|
| `faiss.index` | dense vectors, `IndexFlatIP` |
| `bm25.pkl` | `{bm25, doc_ids, tokenizer_note, …}` — `doc_ids` IS the score-array order |
| `docmap.json` | FAISS row ordinal ↔ docid ↔ raw text |
| `index_meta.json` | every decision below, plus timings, VRAM and versions |

Code: `src/common.py` (shared invariants), `01_index.py` (build), `verify_stage1.py` (checks).

---

## 0. How to run it (Colab)

Upload `01_index.py`, `verify_stage1.py` and `src/common.py` (keep `common.py` inside a
`src/` folder) into the project root on Drive, alongside the Stage 0 files. Then, after
Stage 0's install + restart:

```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT     # so src/common.py resolves paths to Drive
os.chdir(PROJECT)
```

```python
!python 01_index.py
```

```python
!python verify_stage1.py
```

`SLOTA_ROOT` is what makes the paths point at Drive rather than at the VM's ephemeral disk —
without it a session timeout costs the whole embed. Re-runs of `01_index.py` are no-ops;
pass `--force` to rebuild. `verify_stage1.py` must be a separate `!python` invocation, not a
notebook cell reusing Stage 1's variables — the cold-reload path is the thing under test.

---

## 1. Corpus

- **SciFact via BEIR**, `split="test"`, downloaded from the BEIR mirror and cached at
  `artifacts/data/scifact/`. `load_scifact()` skips the download entirely when
  `corpus.jsonl` is already there, so re-runs and later stages need no network.
- Chosen because it ships ground-truth qrels — every recall/nDCG figure later is real
  rather than hand-annotated (methodology §1).
- Sanity target: **5183 docs / 300 test queries**. These are published BEIR stats used
  as a *check*, never as a number we report.

**Text field — one canonical string, used by BOTH arms:**

```python
doc_text(doc) = (title + " " + text).strip()
```

This is the BEIR baseline convention. Both arms consume exactly this string, so the
BM25-vs-dense comparison cannot be contaminated by a preprocessing difference. It lives
in `src/common.py` precisely so no stage can drift from it.

**Doc order = the FAISS ordinal space.** `sorted_doc_ids()` sorts numerically when every
docid is a digit string (SciFact's are), else lexicographically. Sorting rather than
trusting dict insertion order makes the ordinal↔docid map reproducible across runs and
machines. This ordering is *also* BM25's row order, so under one-vector-per-doc both arms
share one ordinal space.

---

## 2. Dense arm

| Setting | Value | Why |
|---|---|---|
| model | `BAAI/bge-base-en-v1.5` | methodology §2 champion for the dense paradigm |
| dtype | **fp16** | T4/sm_75 — bf16 is not natively supported (Stage 0 confirmed) |
| attention | **`sdpa`** | FlashAttention-2 needs sm_80+; §3 forbids it here |
| max_seq_length | 512 | set explicitly, not inherited |
| batch size | 64 | tunable via `--batch-size` |
| passages | **raw text, no prefix** | bge's own convention |
| queries | bge instruction prepended, **at search time only** | see below |

**bge ≠ e5.** Reference repo A embeds with `intfloat/e5-large-v2` and prepends
`"query: "` / `"passage: "` to everything (`reference/hybrid-rag/src/ingest.py:147`,
`retrieval.py:108`). bge-en-v1.5 does not use that scheme: passages are embedded raw, and
retrieval queries optionally take the instruction
`"Represent this sentence for searching relevant passages: "`. Copying repo A's prefixes
onto a bge model would silently degrade both arms. The instruction constant lives in
`src/common.py` and is applied only by `embed_queries()`, which Stage 1's smoke check and
Stage 2 both call — one code path, so the convention cannot diverge between them.

**Normalization — order matters.** `_finalize()` casts the fp16 encoder output to float32
*first*, then calls `faiss.normalize_L2` in place. Casting first means the norm is computed
at full precision; FAISS also refuses anything but contiguous float32. After this, inner
product **is** cosine similarity, which is the entire justification for `IndexFlatIP`.
`01_index.py` asserts `‖v‖ ≈ 1` for every vector and aborts if not — a silently
unnormalized index would turn every cosine into a magnitude-biased dot product.

**Why `IndexFlatIP` and not ANN.** At ~5K docs, exact exhaustive search is milliseconds
*and* exact. An approximate index would drop true neighbours; that loss would surface as
reduced recall, and we would then be unable to separate *"dense retrieval underperformed on
scientific text"* from *"the index lost the answer"*. Exactness here is a
**measurement-validity** decision, not a performance one. No server, no Docker, no
`faiss-gpu`.

---

## 3. Sparse arm

- `rank_bm25.BM25Okapi`, library-default `k1`/`b`.
- **Tokenizer: `str.lower().split()`** — repo A's convention, kept deliberately
  (`notes/repo-read.md` §5): no stemming, no stopword removal, punctuation retained. This
  is plausibly *favourable* on SciFact, where claim↔evidence term overlap is high and
  scientific terms survive better unstemmed — but it is a variable we own, so it is stated
  rather than assumed. Swapping it later would be a legitimate expand-on-demand arm,
  triggered by a result.
- **Doc-level, unchunked, untruncated.** BM25 has no input length limit, so it sees the
  full text even for docs the dense encoder had to truncate.
- The pickle stores `doc_ids` alongside the model because `get_scores()` returns an array
  positionally aligned to build order — `scores[j]` is `doc_ids[j]`. Losing that mapping
  would make every sparse result garbage in a way no exception would catch.
- The pickle is tied to the `rank_bm25` version, which is recorded in `index_meta.json`;
  rebuild if that version changes.

---

## 4. Chunk→doc policy (the load-bearing decision)

`01_index.py` measures the corpus's token-length distribution under **bge's own tokenizer**
(with special tokens) and then decides. Three branches:

1. **No doc exceeds 512** → `one_vector_per_doc`, `dedup_needed=false`. Nothing truncated,
   nothing chunked. Ideal.
2. **≤ `TRUNCATION_TOLERANCE` (1%) exceed 512** → still `one_vector_per_doc`; the encoder
   truncates those docs at 512, and every affected docid is recorded in
   `index_meta.json → truncation.truncated_docids`.
3. **More than 1% exceed 512** → `chunked`, `dedup_needed=true`, and Stage 2 **must**
   collapse chunks to doc level before computing anything against doc-level qrels, or
   recall inflates (`notes/repo-read.md` §4 — repo B's `_unique()` exists for exactly this).

**Why a tolerance rather than the strict "any doc exceeds → chunk" rule.** Chunking imposes
doc-level dedup on every downstream stage permanently. Paying that complexity for a handful
of docs out of ~5183 would be a bad trade, *provided the cost of truncation is measured
rather than assumed* — which is what the over-limit audit does. Override with
`--chunk-policy {one_vector_per_doc,chunked}`.

### MEASURED OUTCOME: branch 3, `chunked` — and the threshold was the wrong instrument

The distribution is far heavier than a title-plus-abstract corpus suggests:
median 316, **p95 567, p99 760, max 1939**, with **455/5183 docs (8.78%)** over the 512-token
window. That is 8.8× the tolerance, so `auto` selected `chunked`. It produced 5661 units —
933 chunks from the 455 long docs, 2.05 chunks each on average.

The 1% threshold got the right answer for the wrong reason, and the real reason is worth
recording because it inverts the tradeoff as originally written:

**SciFact abstracts are structured — BACKGROUND / METHODS / RESULTS / CONCLUSIONS — and
claim evidence concentrates in RESULTS and CONCLUSIONS, i.e. the tail.** Truncating a
1939-token abstract at 512 keeps the background and discards the evidence, systematically,
in a known direction. Worse, it would mean the dense arm sees 512 tokens of a document while
BM25 sees all 1939 — so **truncation is what breaks parity between the arms, and chunking is
what restores it.** Both arms now see all the text, which is the precondition for the
BM25-vs-dense comparison being about paradigms rather than about input budgets.

So the complexity is bought deliberately. What we pay for it is in §8 below.

**The audit that quantifies the alternative:** Stage 1 intersects the over-limit docids with
the gold docids in qrels and reports how many qrels pairs truncation would have put at risk
(`index_meta.json → truncation.n_over_limit_gold` and `.qrels_pairs_at_risk_if_truncated`).
This runs under *both* policies, so the record shows what truncating would have cost, not
just what chunking cost. Under `one_vector_per_doc`, zero gold docs over the limit would mean
truncation provably cannot move any metric.

**Asymmetry, stated:** truncation would apply to the dense arm only; BM25 always sees full
text. Recorded in `index_meta.json`.

**Chunker choice, if branch 3 triggers:** a fixed token-window chunker over the *embedder's*
tokenizer, with 64-token overlap — deliberately **not** repo A's `chunk_text()`
(`reference/hybrid-rag/src/ingest.py:27`). Repo A splits on blank-line paragraphs then
sentences, and counts with tiktoken's `cl100k_base`. SciFact docs are a title plus one
unbroken abstract: there are no paragraphs to split on, and cl100k counts are not what
bge's BERT tokenizer will do, so a "512-token" chunk could still overflow the encoder.
Windowing on the model's own tokenizer is the only way to guarantee no chunk overflows.

---

## 5. Divergences from repo A's `ingest.py`

| Repo A | Here | Consequence if we had copied it |
|---|---|---|
| Qdrant collection, HTTP server | FAISS `IndexFlatIP`, in-process | needs Docker; ANN recall error contaminates the metric |
| `intfloat/e5-large-v2` + `query:`/`passage:` prefixes | `bge-base-en-v1.5`, raw passages + bge query instruction | wrong prefix scheme silently degrades embeddings |
| `uuid.uuid4()` per chunk as the doc id | **BEIR docids preserved verbatim** | fresh UUIDs would destroy the docid↔qrels linkage — every metric would be zero, or worse, meaningless |
| paragraph/sentence chunker, tiktoken counts | one vector per doc; token-window chunker only if forced | wrong tokenizer → chunks that overflow the encoder |
| walks `*.txt` files off disk | BEIR loader | n/a |
| writes `corpus.jsonl`, no persisted vectors | persists FAISS + BM25 + docmap + meta | re-embedding the corpus on every run |

The UUID row is the dangerous one: repo A's ids are generated at ingest time, so nothing in
its pipeline can ever be scored against an external ground truth. Preserving BEIR docids is
what makes the qrels usable at all.

---

## 6. Idempotency

`01_index.py` exits early, printing the existing artifact set and its meta, if all four
files are present. `--force` rebuilds. Embedding the corpus is the one non-trivial compute
cost in Phase 1; nothing should pay it twice (methodology §9 step 1).

---

## 7. Verification

Checks 1, 2 and 6 run inside `01_index.py`; checks 3, 4 and 5 run in `verify_stage1.py`,
which shares **no in-memory state** — it reloads everything cold from disk, because that is
what every later stage does.

The one worth calling out is the **alignment proof** in check 4. Asserting
`ntotal == n_docs` and that ordinals round-trip through the docmap does *not* catch a
shuffled map — a permuted `ordinal_to_docid` satisfies both. So `verify_stage1.py` also
pulls the stored vector back out of FAISS (`index.reconstruct(ordinal)`), re-embeds that
docid's text, and checks `cos(stored, re-embedded) ≥ 0.99`. It then runs a **negative
control** — the same stored vector against a *different* doc's text, which must score
clearly below 1.0 — so that a comparison bug returning 1.0 unconditionally cannot look like
a pass. This is the check that would catch the "wrong ordinal↔docid map silently corrupts
every dense result" failure. (Skipped under `chunked`, where docmap holds full doc text
rather than chunk text.)

Note on the tolerance: fp16 encoding is not bit-exact across differing batch shapes, so the
re-embedded vector is compared with a 0.99 cosine floor rather than for exact equality.

### Results — run of 2026-08-05 (raw log: `notes/stage1-output.md`)

| Check | Expected | Measured | |
|---|---|---|---|
| 1. corpus / queries / qrels | 5183 / 300 / ~339 | **5183 / 300 / 339**, 283 distinct gold docs, 23 multi-gold queries | PASS |
| chunk_policy branch | `one_vector_per_doc` hoped | **`chunked`**, `dedup_needed=true` | see §4 |
| token lengths | short abstracts | min 70, median 316, mean 337, p95 567, p99 760, **max 1939** | — |
| docs over 512 | small tail | **455 (8.78%)** → 5661 units, 933 chunks from 455 docs | — |
| over-limit docs that are gold | wanted | **46 of 283 gold docs (16.3%)**; 53 of 339 qrels pairs would have been at risk under truncation | see below |
| chunks | — | 933 chunks from 455 docs, mean 2.05, **max 5** (matches window 510 / step 446 on the 1939-token doc) | PASS |
| 2. FAISS `ntotal` / `d` | = units / 768 | **5661 / 768**; file 16.6 MiB = 5661×768×4 B exactly | PASS |
| L2 norms | ≈ 1.0 | 1.000000 / 1.000000 → IP == cosine | PASS |
| 3. cold reload | all four load | FAISS, BM25 (rank_bm25 0.2.2), docmap, meta all OK | PASS |
| 4. round-trip 3 docids | both directions | 4983→ord 0/row 0; 16172576→ord 2830/row 2595; 198309074→ord 5660/row 5182 | PASS |
| 4. alignment cosine | ≥ 0.99 + control | **1.000000** on all 3 probed ordinals; negative control **0.620322** | PASS |
| 5. smoke retrieval | all hits resolve | 5 dense + 5 BM25, all `resolves=True`; 0/5 overlap; gold in neither | PASS (wiring) |
| 6. peak VRAM | ≪ 14.5 GiB | **1.07 GiB reserved / 0.82 GiB allocated** | PASS |
| 6. wall-clock | ~a minute | embed **31.9s** (178 units/s), total **69.0s** | PASS |

### Truncation's measured cost — why chunking was the right buy

`over_limit ∩ gold` is the decisive number: **46 of 283 gold docs (16.3%) exceed the window,
against 8.78% corpus-wide.** Gold documents are over-long at nearly double the base rate. That
is the tail-evidence argument appearing in the data rather than in an argument: claims draw
their evidence from papers with detailed structured abstracts, and those are the long ones.
Truncating would have cut text from 46 of the documents the metric depends on and endangered
**53 of 339 qrels pairs (15.6%)**. The chunking complexity is bought with evidence.

### Query↔doc cosine asymmetry — carry into Stage 2

The negative control (two unrelated abstracts) scored **0.620**, *higher* than the best dense
hit for the smoke query (**0.582**). Expected bge behaviour — doc↔doc cosines run
systematically above query↔doc cosines, since queries are short, stylistically unlike
passages, and carry the instruction prefix. Two consequences:

- **Absolute cosine is not a relevance threshold.** Only within-query rank is meaningful. This
  is the same property that makes RRF rank-only, so it is a §8 checklist point, not trivia.
- The top-5 spread is flat (0.5819 → 0.5626, range 0.019) at a low absolute level, which makes
  the **bge query instruction a worthwhile Stage 2 variable**. bge-v1.5's own documentation
  calls it optional; testing both costs a re-embed of 300 queries and no corpus re-embed.

### The gap that was closed, and the one after it

The **alignment proof did not run.** It was written to skip under `chunked` because
`docmap.json` held only full doc text, and a chunk's vector cannot be checked against the
whole document's text. That skipped exactly the check designed to catch a shuffled
ordinal↔docid map — the failure mode that corrupts every dense result silently and that no
other check in this list would catch (a permuted `ordinal_to_docid` satisfies every shape
assertion and every round-trip).

Soft evidence that alignment is in fact fine: the dense top-5 for *"0-dimensional
biomaterials show inductive properties"* is cortical bone fracture, a graphene iPSC culture
platform, nonlinear elasticity in biological gels, a cytotoxicity assay, and nanotoxicology —
a coherent biomaterials neighbourhood. A shuffled map would return topical noise. But that is
inference, not proof.

Patch applied to Stage 1:

1. `docmap.json` now stores **`ordinal_to_text`** — the exact string embedded at each ordinal
   (the decoded chunk under `chunked`). The alignment proof needs it; Stage 3 will need it too,
   since a reranker must score the text that was actually retrieved.
2. `verify_stage1.py` runs the proof under **both** policies, over **every ordinal** of each
   probe docid rather than just the first — under chunking a doc owns several rows and any one
   of them could be misplaced. The negative control now also fails loudly instead of only
   printing.
3. The over-limit ∩ gold audit runs under **both** policies, and additionally reports how many
   qrels pairs truncation would have endangered.
4. `chunking` stats block added to `index_meta.json`; the chunker's tokenizer warning silenced
   (it was correct behaviour that read as a defect in the log).

Re-run of 2026-08-05 confirmed all four: alignment 1.000000, control 0.620322, gold audit
present, docmap 15.3 MiB.

**Residual gap found in that run:** all three probe docids (`4983`, `16172576`, `198309074`)
have exactly one ordinal, so the "every ordinal of each probe docid" logic never fired and the
failure mode chunking *specifically* introduces — a doc's 2nd or 3rd vector landing at the
wrong ordinal — remained untested. Probes are chosen by position (0, middle, last) and none
happened to be split.

`verify_stage1.py` now forces multi-chunk docs into the probe set under `chunked` policy: the
deepest doc (most chunks) plus a 2-chunk doc, and it asserts each split doc's ordinals are
**contiguous**, since `build_units()` appends a doc's chunks consecutively and a gap would mean
the ordinal space and chunk order had come apart. Verifier-only change — no index rebuild.

Stage 1 is signed off once that re-run shows the multi-chunk probes passing.

---

## 8. What Stage 1 hands Stage 2 as obligations

Consequences of the `chunked` outcome. These are not optional.

1. **Collapse chunks to doc level BEFORE any metric.** 5661 retrieval units map to 5183 docs;
   qrels are doc-level. Scoring at chunk level inflates recall (`notes/repo-read.md` §4 —
   repo B's `_unique()` exists for precisely this). `docmap.dedup_needed` is `true`.
2. **The collapse rule is a stated variable.** Max-over-chunks is the standard choice and the
   default to take, but it applies to the dense arm only — BM25 needs no equivalent. Record it
   in the Stage 2 meta the way the tokenizer choice is recorded here. A doc whose evidence is
   split across two chunks may score differently under max vs. mean; do not leave the choice
   implicit.
3. **RRF operates on doc-level rank lists, after collapse** — not on chunk ordinals. Fusing
   chunk ranks would let one long document occupy several slots and dominate the fused list.
4. **Watch the BM25 tokenizer.** The smoke query tokenizes to
   `['0-dimensional', 'biomaterials', 'show', 'inductive', 'properties.']` — note `properties.`
   with the period welded on, which matches only where a sentence happens to end in that word.
   Vocabulary of 87,379 types over ~1.1M tokens is consistent with punctuation inflating the
   type count. BM25's top-5 on that query was entirely epithelial-mesenchymal-transition
   papers, topically unrelated to the claim. This is the deliberate variable from
   `repo-read.md` §5 showing its cost. If BM25 underperforms in Stage 2, a punctuation-stripping
   tokenizer is the first evidence-triggered expansion arm to add — and per methodology §2.1,
   the trigger has to be the result, not a hunch.
5. **Sanity band, not a target.** Published SciFact figures put BM25 nDCG@10 near 0.66 and
   bge-base near 0.74. Numbers near zero mean a wiring bug; numbers in that neighbourhood mean
   the 0/5 smoke result was just a hard query. We report only our own measurements
   (methodology §8: no figure copied from any leaderboard) — this band is a bug detector.

This table plus `index_meta.json` is the Stage 1 audit trail; the §8 checklist item "every
number came from a JSON file in `artifacts/`, traceable to the stage that made it" starts here.

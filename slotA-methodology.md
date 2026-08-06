# Slot A — Retrieval + Evaluation Harness: Methodology

**Status:** Planning artifact. No code written yet. This document defines the benchmark
table (the deliverable) before any implementation, per the roadmap's "measurement plan
comes first" rule.

**Narrative claim this project earns:** *"I don't just build RAG — I measure it, at the
retrieval, reranking, and generation layers, and I know which intervention moved which
number and by how much."* Almost no student resume can say the middle clause.

---

## 1. Scope (locked)

**In:** retrieval-quality metrics, reranking uplift, generation faithfulness — as a single
before/after intervention table on a standard labeled benchmark.

**Out (explicitly not this project):** ingestion/parsing eval, filter false-exclusion,
ontology-grounded eval, production monitoring/drift, A/B testing. These exist in the
reference material and are the main way this project bloats past 15 days. Do not build them.

**Corpus:** SciFact (BEIR). Chosen because it ships ground-truth relevance judgments — every
recall@k / nDCG number is real, not hand-labeled by you. Removes the annotation burden that
would consume the timeline.

---

## 2. The experiment: a factorial grid, NOT a pre-chosen pipeline

**Design principle (load-bearing):** an eval harness must not pre-select its own pipeline. The
deliverable is the *evidence for* the choice, not a choice with metrics bolted on afterward.
Therefore we measure the cross-product of retriever × reranker and read the winner off the grid.

**Do NOT assume hybrid/RRF or any reranker is best.** SciFact is dense scientific text with heavy
claim↔evidence term overlap — BM25 is a known-strong baseline here and may beat dense retrieval
outright. If it does, that measured result is a STRONGER bullet than any reranker win ("measured
lexical vs. semantic on a scientific-claim corpus; the simpler BM25 baseline won, against the
prevailing assumption").

### The grid (9 cells to start — one champion per category)

**Design principle:** one strong, current REPRESENTATIVE per retrieval paradigm — not the whole
model family. The grid answers "does lexical beat semantic beat fusion, on THIS corpus?" — a
paradigm-vs-paradigm question that one champion per side answers cleanly. Running whole families
is a different, narrower question ("model X vs. model Y") and is deferred.

| Axis | Champion (1 each) | Question it answers |
|---|---|---|
| Sparse/lexical | BM25 | Does keyword matching suffice on scientific text? |
| Dense/semantic | bge-base-en-v1.5 | Does meaning-based search beat keywords here? |
| Fusion | RRF hybrid | Does combining beat either alone? |
| Cross-encoder rerank | bge-v2-m3 | Does a classic reranker help? |
| LLM-based rerank | bge-v2-gemma | Does the frontier reranker help more, and at what cost? |

| retriever ↓ / reranker → | none | bge-v2-m3 (cross-enc) | bge-v2-gemma (LLM) |
|---|---|---|---|
| **BM25**       | · | · | · |
| **Dense**      | · | · | · |
| **RRF hybrid** | · | · | · |

**Do NOT assume hybrid/RRF or any reranker is best.** SciFact is dense scientific text with heavy
claim↔evidence term overlap — BM25 is a known-strong baseline here and may beat dense outright.
If it does, that measured result is a STRONGER bullet than any reranker win.

### The expand-on-demand rule (this is what makes it rigor, not flailing)

Widen an axis ONLY when a Phase-1 result raises a specific question the current grid can't answer.
The expansion is *triggered by evidence*, never planned up front. Example: if BM25 beats dense,
THAT licenses adding a second dense model — to test whether the whole dense *paradigm* underperforms
here, or just that one embedding model. A second dense arm is cheap (re-embed + re-retrieve, no
generator) and only added if Phase 1 leaves time. "Exhaustiveness is not rigor" — ten configs you
can each explain beat fifty you can't.

### Two-phase execution (this is what keeps the factorial from exploding)

```
PHASE 1 — CHEAP, no generator.  All 9 cells, RETRIEVAL + LATENCY metrics only.
          recall@k, MRR, nDCG@k need only SciFact's ground-truth judgments.
          → build the quality-vs-latency frontier → pick top 2–3 survivors.

PHASE 2 — EXPENSIVE, GPU.  Only the 2–3 survivors get generation + faithfulness.
          Never spend the costly measurement on configs the cheap metrics already killed.
```

Reranking is order-independent of the retriever's identity — it only sees the candidate set. So
Phase 1 is: 3 retrieval runs → save candidates → rerank each set 3 ways. Cheap relative to generation.

### The full pipeline (for the surviving configs only)

```
retriever → top-50 → reranker → top-k → local generator → faithfulness judge
```

**Role of the local generator (Qwen2.5-1.5B):** it exists SOLELY to produce the answer text whose
faithfulness is then scored. Faithfulness is a property of *generated text* — no generation, no
faithfulness axis. If this project were retrieval-only, the generator (and the entire VRAM
problem) would not exist. It is the price of the faithfulness axis, which was chosen because it
hardens Prodigal.

**The LLM-as-judge you now must defend:** faithfulness scored by a judge model means the JUDGE'S
OWN reliability is an interview question. Mitigation (non-optional, budget for it early):
- Hand-label a small set (30–50 answers) yourself: faithful / unfaithful.
- Report judge-vs-human agreement (Cohen's κ or QWK) on that set.
- This is the SAME machinery behind Prodigal's QWK 0.82 — which is precisely why this project
  hardens Prodigal. Do not discover this requirement late; it is part of the deliverable.

**generator:** Qwen2.5-1.5B-Instruct on the T4 (the 2nd heavy resident — see §3). Low reasoning
load: evidence is in-context, answers are short and grounded. A test subject, not a product — an
occasional weak/unfaithful answer is fine (arguably useful, since it gives the judge something
real to catch).
**judge:** Qwen2.5-7B-Instruct. ~14GB fp16 — fits because the judge runs ALONE in its stage
(generator and reranker already unloaded, whole T4 available); 4-bit if tight, and if quantized,
the bullet says so. Chosen as the strong, current, well-supported default with reliable
structured output. **Its scores are only trusted after validation against the hand-labeled set
below** — if agreement is poor, spot-check close calls with a frontier API model.
faithfulness = supported_claims / total_claims (claim-extract → entailment judge → aggregate).
**Dense retriever:** BAAI/bge-base-en-v1.5 (embed once at index time, cheap, T4-fine). Use
bge-small if the surviving-config generation stage is VRAM-tight.

---

## 3. GPU reality (T4, 16GB, fp16, sm_75)

**The binding constraint:** bge-reranker-v2-gemma (2.5B, ~5GB fp16) in Stage 3, generator
Qwen2.5-1.5B (~3GB) in Stage 4, and judge Qwen2.5-7B (~14GB fp16) in Stage 5 cannot be resident
together. Staged to disk, none ever coexist — each stage loads one model, runs, unloads. The 7B
judge is the tightest single resident and gets the whole T4 to itself; 4-bit if needed.

**Note this is a CHOSEN constraint, not a law.** The second heavy task exists only because you
chose a LOCAL generator (for self-containment/defensibility). An API generator would leave the
GPU entirely, and bge-gemma would be the only VRAM pressure. Staging-to-disk is the price of the
local-generator choice — a fair trade, taken knowingly.

### The staged-to-disk architecture (non-negotiable)

Each stage is a separate script/notebook that reads the previous stage's output file and
writes its own. No two large models ever resident at once.

```
01_index.py     bge-base→FAISS flat; rank_bm25→BM25       [adapt A/ingest.py]     → artifacts/index/{faiss.index,bm25.pkl}
02_retrieve.py  load both indexes → top-50 per arm        [reuse A/retrieval.py]  → artifacts/runs/retrieval_{arm}.json
03_rerank.py    load ONE reranker → rerank → unload       [rewrite A/rerank.py]   → artifacts/runs/rerank_{model}.json
04_generate.py  load Qwen-1.5B → answer → unload          [adapt A/generate.py]   → artifacts/answers/{arm}.json
05_judge.py     load Qwen-7B → faithfulness → validate    [adapt A/judge.py]      → artifacts/scores/faithfulness_{arm}.json
06_report.py    assemble grid + Pareto frontier           [write from scratch]    → artifacts/report/table.md
```

**Why this shape:**
- Survives Colab session timeouts — restart at any stage, prior outputs persist to Drive.
- bge-gemma reranking (03) fully completes and unloads before the generator (04) loads.
- Every number is traceable to the JSON file that produced it — this is your interview defense.

### VRAM guardrails
- **No FlashAttention-2** (needs Ampere sm_80+). Use PyTorch SDPA. Set `attn_implementation="sdpa"`.
- **fp16 only**, no bf16.
- **Checkpoint to Drive** after each stage.
- **Verify GPU allocation at session start** — free-tier assignment varies; a K80/P100 day changes the plan.
- If bge-gemma OOMs even staged: load it in **4-bit** (bitsandbytes) — but note that quantizing
  the reranker is itself a measurable variable; if you go 4-bit, SAY SO in the bullet, don't hide it.

---

## 4. The deliverable: a quality-vs-latency FRONTIER (not a leaderboard)

The winning system is NOT "highest nDCG." It is **best quality per unit latency** — the config at
the knee of the curve, where most of the quality gain is captured before latency explodes. This
framing is the point: it shows deployment-cost reasoning, which almost no student resume does.

**Phase 1 table — all 9 cells, quality + latency, no generation:**

| Config (retriever → reranker) | Recall@10 | MRR | nDCG@10 | P@1 | Latency/query (ms) |
|---|---|---|---|---|---|
| BM25 → none               | · | · | · | · | · |
| BM25 → bge-v2-m3          | · | · | · | · | · |
| BM25 → bge-v2-gemma       | · | · | · | · | · |
| Dense → none              | · | · | · | · | · |
| Dense → bge-v2-m3         | · | · | · | · | · |
| Dense → bge-v2-gemma      | · | · | · | · | · |
| RRF hybrid → none         | · | · | · | · | · |
| RRF hybrid → bge-v2-m3    | · | · | · | · | · |
| RRF hybrid → bge-v2-gemma | · | · | · | · | · |

**Reading the grid:**
- A config that is *both* lower-quality AND slower than another is **dominated** — discard it.
- The survivors form the **Pareto frontier**: each is the best quality at its latency budget.
- The recommended system is the **knee** — the frontier point past which extra latency buys
  little quality. That is the defensible "I chose X" answer.

**Phase 2 table — survivors only, adds the generation axis:**

| Surviving config | nDCG@10 | Latency/query | Faithfulness |
|---|---|---|---|
| (top 2–3 from Phase 1) | · | · | · |

**Resume bullets come from the frontier, not single cells:**
- The tradeoff finding: which config is the knee, and what the "premium" configs cost per nDCG point
- BM25-vs-dense on a scientific corpus (the "simpler won" result, if it holds)
- **LLM-reranker vs. cross-encoder** — the 2026 differentiator, stated WITH its latency multiplier
- faithfulness conditioned on retrieval quality — the Prodigal-hardening line

**Every quality gain is reported with its latency cost.** A reranker adding 8 nDCG at 40× latency
is a *finding*, not a win — and the tradeoff framing is what interviewers respect.

---

## 5. Metric definitions (rebuild these from memory — the authenticity bar)

These are ~15 lines each. If you can't rewrite them without looking, you haven't earned the bullet.

- **Recall@k** = |relevant ∩ top-k| / |relevant|
- **MRR** = mean over queries of 1/(rank of first relevant doc)
- **nDCG@k** = DCG@k / IDCG@k, DCG@k = Σ (2^rel_i − 1)/log2(i+1)
- **RRF** = Σ_lanes 1/(k + rank), k=60 canonical; rank-only, ignores raw scores (this is WHY it
  works across incomparable BM25/cosine scales — a good interview talking point)
- **Faithfulness** = supported_claims / total_claims; structure (which claims failed) matters
  more than the scalar

Use `ranx` or `pytrec_eval` for the *production* numbers (they handle graded relevance and TREC
edge cases correctly), but keep your hand-written versions as the unit test that proves you
understand them. The reference repo does exactly this — pinned worked examples as `pytest` tests.

---

## 6. Reference repos — clone both, study, don't run-to-completion

Two repos, two jobs. Neither is clone-and-run for your design; each supplies different pieces.
The authenticity line holds: **every number in your final table is one you measured on your own
hardware.** Reference numbers (their READMEs, their gold-set correlations) are never copied.

### Repo A — pipeline skeleton + judge rubric: `tim-ponomarev/hybrid-rag`
`git clone https://github.com/tim-ponomarev/hybrid-rag`

Closest ARCHITECTURAL match. Its pipeline is nearly yours: BM25 + dense (top-50) → RRF →
cross-encoder rerank (top-10) → LLM answer → LLM-as-judge (faithfulness / relevance / coverage).
Modules are already separated the way your stages need. **Caveat:** 0 stars, single author,
v0.1.0, unvetted — read every file before running; trust none of its numbers.

Its design differs from yours in three ways you deliberately change (each change is a headline
contribution, not a copy):
- Judge = OpenAI `gpt-4o-mini` (API) → **you replace with local Qwen2.5-7B.**
- Reranker = `ms-marco-MiniLM-L-6` (old) → **you upgrade to bge-v2-m3 + gemma arm.**
- 4 hand-picked rows on LegalBench → **you rebuild as the 3×3 SciFact grid + Pareto frontier.**

### Repo B — SciFact wiring + metric-correctness tests: `slavadubrov/rag-evals-demo`
`git clone https://github.com/slavadubrov/rag-evals-demo`

Closest CONTENT match for retrieval metrics on SciFact specifically. Notebook 01 sweeps
Recall@k / MRR / nDCG over a real SciFact index and **pins known-good values as unit tests**
(Recall@5 = 0.750, MRR = 0.625, nDCG@5 = 0.627 in `tests/test_retrieval_metrics.py`). Use these
to verify YOUR hand-written metric code is correct before you trust any grid cell — the
"rewrite from memory, check against known-good" loop the authenticity checklist demands.

### Per-module reuse map (what's pre-built vs. what you write)

| Your stage | Pre-built source | Reuse level | What you change |
|---|---|---|---|
| `01_index.py` | A `src/ingest.py` | **Adapt** | Point at SciFact (BEIR loader); replace Qdrant with FAISS `IndexFlatIP` (exact, no server) + `rank_bm25`; keep chunking logic |
| `02_retrieve.py` | A `src/retrieval.py` | **Reuse near-whole** | BM25 + dense + RRF already correct here; add latency timing + JSON dump per arm |
| metric math | B notebook 01 / `evaluation/retrieval.py` | **Rewrite, verify** | Hand-write recall@k/MRR/nDCG; check against B's pinned unit-test values |
| `03_rerank.py` | A `src/rerank.py` | **Rewrite core** | A only does one old cross-encoder; you write a model-swappable reranker running m3 AND gemma, staged+unloaded |
| `04_generate.py` | A `src/generate.py` | **Adapt** | Swap API/OpenAI call → local Qwen2.5-1.5B via transformers; keep citation-prompt structure |
| `05_judge.py` | A `src/judge.py` + `judge.py` | **Adapt heavily** | Keep the faithfulness/relevance/coverage RUBRIC; swap gpt-4o-mini → local Qwen2.5-7B; add your hand-label validation harness |
| `06_report.py` | — (neither) | **Write from scratch** | The 3×3 grid assembly + Pareto-frontier selection — this is wholly yours |

**Legend:** *Reuse near-whole* = runs with config changes. *Adapt* = same shape, swap components.
*Rewrite core* = keep the interface, rebuild the logic (because your version does more).
*Write from scratch* = no reference exists; this is your original contribution.

**The load-bearing point:** the two stages that are most "yours" — `03_rerank.py` (multi-model
staged reranking) and `06_report.py` (factorial grid + frontier) — are exactly the two that
carry your headline results. The reused stages (index, retrieve) are plumbing. That distribution
is correct: reuse the plumbing, build the parts you'll be interviewed on.

---

## 7. Resume framing (drafted early so the build targets the right numbers)

Do NOT write final bullets until numbers exist. But build toward these shapes so you measure
the right things:

- Retrieval delta bullet: hybrid/RRF vs. single-lane, recall@k + nDCG, on N SciFact queries
- Reranker bullet: **LLM-based vs. cross-encoder**, ΔnDCG at stated latency cost
- Generation bullet: faithfulness lift from better retrieval — the Prodigal-hardening line
- Systems bullet: the staged pipeline surviving the T4 constraint (this is real engineering,
  not just metrics — it shows you can ship under a hardware ceiling)

**Prodigal cross-link:** this project builds the exact LLM-as-judge validation machinery
Prodigal's scorer leans on (Prodigal reports QWK 0.82 but has no retrieval underneath it). On
the resume, frame Slot A as the *eval layer* and Prodigal as the *application* — distinct layers,
mutually reinforcing, never redundant.

---

## 8. Authenticity checklist (gate before any bullet ships)

- [ ] Every number came from a JSON file in `artifacts/`, traceable to the stage that made it
- [ ] I can rewrite recall@k, MRR, nDCG, RRF from memory
- [ ] I can explain why RRF uses rank not score
- [ ] I can explain why bge-gemma beat (or didn't beat) bge-v2-m3 on THIS corpus
- [ ] I validated the faithfulness judge against a hand-labeled set and can state the agreement
- [ ] If I quantized anything, the bullet says so
- [ ] No figure is copied from the reference repo's README or any leaderboard
- [ ] I know the latency cost of every quality gain in the table
- [ ] Every axis I widened was triggered by a result, and I can name the result that triggered it

---

## 9. Build order (next session)

**Step 0 — clone both repos, read before writing.**
`git clone` A (`tim-ponomarev/hybrid-rag`) and B (`slavadubrov/rag-evals-demo`). Read A's `src/`
end to end — it's unvetted (0 stars, v0.1.0), so trust the *structure*, verify the *logic*. Run
A's `scripts/smoke_test.py` (in-memory, no server/API/keys, ~5s) to confirm the RRF merge works
before you touch anything.

**PHASE 1 — the 9-cell grid, cheap, no generator:**
1. `01_index.py` — **adapt A's `ingest.py`. Builds both indexes ONCE and persists them to disk;
   every later stage loads them instead of rebuilding — saving time and compute on every re-run.**
   Repoint to SciFact via the BEIR loader; keep A's chunking. Two parallel indexes, one per arm:
   - **Dense (FAISS):** embed the full corpus once with `BAAI/bge-base-en-v1.5` (bi-encoder) →
     store vectors in a FAISS `IndexFlatIP` (exact, exhaustive) → save to `artifacts/index/faiss.index`.
   - **Sparse (BM25):** tokenize raw corpus text → build a `rank_bm25` index → pickle to
     `artifacts/index/bm25.pkl` (plus the docid↔text mapping for both arms).

   Embedding the corpus is the one non-trivial compute cost here; persisting it means you pay it
   ONCE, not on every retrieval/rerank/report re-run. At SciFact's ~5K docs, exact brute-force
   search is milliseconds AND correct; an ANN index (Qdrant HNSW) would inject recall error into
   the very retrieval metric you're measuring — you could not separate "dense underperformed" from
   "the index dropped true neighbors." No server, no Docker. Verify GPU first. (Interview line:
   "exact search, because at 5K docs a vector DB's approximation would contaminate the metric.")
2. `02_retrieve.py` — **reuse A's `retrieval.py` near-whole** (BM25 + dense + RRF are correct there).
   **Loads the persisted FAISS + BM25 indexes from `artifacts/index/` — no corpus re-embedding;
   only each query is embedded at search time.** Add per-arm latency timing and JSON dump.
   Hand-write the metric functions and **verify them against B's pinned unit-test values** before
   trusting output. **Get the 3-row retrieval table before touching a reranker.**
3. `03_rerank.py` — **rewrite A's `rerank.py` core.** A does one old cross-encoder; you need a
   model-swappable reranker that runs none / bge-v2-m3 / bge-v2-gemma, loading ONE at a time and
   unloading between (gemma 4-bit if VRAM needs it). → completes the 9-cell grid.
4. `06a_frontier.py` — **write from scratch.** Build the quality-vs-latency frontier; drop
   dominated configs. **Pick the top 2–3 survivors.**

**PHASE 2 — survivors only, expensive:**
5. `04_generate.py` — **adapt A's `generate.py`:** swap its OpenAI call for local Qwen2.5-1.5B
   (transformers), keep its citation-prompt structure. Then `05_judge.py` — **adapt A's `judge.py`:**
   keep the faithfulness/relevance/coverage rubric, swap gpt-4o-mini → local Qwen2.5-7B, and add
   the hand-label validation harness (30–50 labels → report agreement). Survivors ONLY.
6. `06b_report.py` — **write from scratch.** Assemble both tables. Fill resume bullets from real cells.

**If GPU is flaky on the day:** steps 0–2 are CPU/light — do them regardless. Defer the gemma
column and report the grid with BM25/dense/hybrid × none/m3. That is already a complete,
defensible project; the gemma arm and Phase 2 are additive.

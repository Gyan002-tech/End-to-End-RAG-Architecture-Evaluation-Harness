# Stage 0 in Colab — upload list and exact run order

Nothing in this project is executed on the local machine. Colab is the runtime.
This file is the complete Stage 0 run sheet.

---

## What to upload

Only three files are needed. Everything else in Stage 0 is either created by the
cells below or cloned fresh from GitHub.

| Upload | Where it goes | Why |
|---|---|---|
| `requirements-colab.txt` | project root | install set part 1 (torch omitted on purpose) |
| `requirements-colab-nodeps.txt` | project root | install set part 2 — `beir` + `FlagEmbedding`, `--no-deps` |
| `env_check.py` | project root | the Stage 0 gate |
| `slotA-methodology.md` | project root | reference while working (optional but useful) |

**Do not upload** `requirements.txt` — it pins `torch==2.5.1`, and reinstalling torch
over Colab's CUDA-matched build is the most common way to break a T4 session. It exists
as the canonical record of the pinned stack, not as a Colab install target.

**Do not upload** `reference/` (~2 MB of git checkouts) — Cell 3 clones both repos.

**Do not upload** `artifacts/` — it is empty scaffolding; Cell 0 recreates it on Drive.

### Runtime setting first

`Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save.`
The whole plan (§3: fp16 only, no bf16, no FlashAttention-2, staged model loading) is
written against a T4/sm_75. If you get an L4 or A100, that is *better hardware but a
changed variable* — `env_check.py` will flag it and the flag has to be honoured.

---

## Cell 0 — Drive mount + project root

Methodology §3 requires checkpointing after every stage so a session timeout never
costs more than one stage. That means `artifacts/` lives on Drive, not on the VM disk.

```python
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
for sub in ['artifacts/index', 'artifacts/runs', 'artifacts/answers',
            'artifacts/scores', 'artifacts/report', 'notes', 'reference']:
    os.makedirs(f'{PROJECT}/{sub}', exist_ok=True)
os.chdir(PROJECT)
print(os.getcwd())
!ls -R artifacts | head -20
```

Then move the two uploaded files into `PROJECT` (drag them into the Drive folder in the
Files pane, or `!cp /content/requirements-colab.txt /content/env_check.py "$PROJECT"/`).

## Cell 1 — what GPU did we actually get

```python
!nvidia-smi
```

Expect `Tesla T4`, `15360MiB`, driver/CUDA in the header. If this says K80 or P100,
stop and re-roll the runtime — those predate the plan's assumptions.

## Cell 2 — install (two parts), then RESTART

Part 1 — the normal resolve:

```python
!pip install -q -r requirements-colab.txt
```

Part 2 — the two packages that must bypass dependency resolution:

```python
!pip install -q --no-deps -r requirements-colab-nodeps.txt
```

`beir` and `FlagEmbedding` are sdist-only with dynamic metadata, and both declare
dependencies we deliberately refuse — an exact `elasticsearch==7.9.1` pin for a retriever
we never import, a duplicate source-built `pytrec_eval`, and FlagEmbedding's training-side
dep tail. Everything either package needs *at runtime* is pinned in part 1.
`requirements-colab-nodeps.txt` documents this in full; Cell 4 section [6] verifies both
actually import, which is the check that makes `--no-deps` safe.

**Then: `Runtime → Restart session`.** This is mandatory, not optional — numpy is already
imported by Colab at startup, so the `numpy==1.26.4` downgrade does not take effect until
the kernel restarts. After restarting, re-run Cell 0 (the `os.chdir`) and continue.

Expect pip to print resolver complaints about preinstalled Colab packages that wanted
numpy 2.x. Those are usually benign; what matters is that Cell 4's import smoke passes.
If `faiss`, `ranx`, `numba`, `pytrec_eval`, `beir` or `FlagEmbedding` fail to **import** in
Cell 4, that is a real problem — send me the traceback.

### If part 1 still fails with `ResolutionImpossible`

Send me the whole error, including the `The conflict is caused by:` block if pip printed
one. To get pip to actually explain itself rather than just naming line numbers, re-run the
failing install verbosely:

```python
!pip install -r requirements-colab.txt 2>&1 | tail -40
```

Dropping `-q` is the point — the quiet flag suppresses the conflict explanation.

## Cell 3 — clone the reference repos (study-only)

```python
!git clone --quiet https://github.com/tim-ponomarev/hybrid-rag reference/hybrid-rag
!git clone --quiet https://github.com/slavadubrov/rag-evals-demo reference/rag-evals-demo
!git -C reference/hybrid-rag log --oneline -1
!git -C reference/rag-evals-demo log --oneline -1
```

I read these at `78ab157` (A) and `5709a4b` (B). If your HEADs differ, tell me — my notes
in `notes/repo-read.md` cite exact line numbers.

## Cell 4 — the environment gate

```python
!python env_check.py
```

Read the `VERDICT` block. Expected on a T4:

- `torch.cuda.is_available()` → `True`
- device name → `Tesla T4`, capability → `sm_75`, VRAM → `~14.7 GiB`
- `native bf16 (cc >= 8.0)?` → `False` ← this is the expected answer; we use fp16
- `torch.cuda.is_bf16_supported()` may still print `True`. Ignore it. Recent torch counts
  *emulated* bf16 on Turing, which is not what §3 means; the script labels it for this reason.
- `empirical fp16 matmul` → `OK`
- `[OK] T4 / sm_75 / fp16 / SDPA` in the verdict

## Cell 5 — repo A's smoke test

```python
!python reference/hybrid-rag/scripts/smoke_test.py
```

Needs only numpy + rank_bm25 (repo A guards its qdrant/sentence-transformers imports),
so it runs with no downloads, no server, no API key, ~5s.

**How to judge the output** — the pass count is *not* the acceptance criterion. The script
uses a hash-stub embedder for the dense lane and says so itself; its own hard gate is only
≥2 of 5. What must be true is that RRF is genuinely fusing:

- each query prints `BM25 top-3`, `Dense top-3`, `RRF top-3`
- `RRF top-3` must be a **merge** — it should contain at least one doc_id absent from
  `BM25 top-3`, and should favour doc_ids appearing in *both* lanes
- if `RRF top-3` equals `BM25 top-3` on every single query, the dense ranking is not
  reaching the fusion step → that is a real failure, report it

Paste the full output back to me and I will confirm the fusion is behaving.

---

## What to send back

1. Cell 1: the `nvidia-smi` device line.
2. Cell 4: the whole `env_check.py` output (especially `VERDICT`, and any `FAILED` import).
3. Cell 5: the whole smoke-test output.
4. Cell 2: any pip error — not the resolver warnings, but anything that actually failed.

Once those three land, Stage 0 is closed and Stage 1 (`01_index.py` — BEIR SciFact →
FAISS `IndexFlatIP` + `rank_bm25`, persisted to `artifacts/index/`) can start.

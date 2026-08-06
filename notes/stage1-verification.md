# Stage 1 Verification & Diagnostic Report

**Status:** VERIFIED & SIGNED OFF  
**Date:** 2026-08-06  
**Target Stage:** Stage 1 (Index Build & PERSIST)

---

## 1. Summary of Verification

The output log from `stage1-output.md` (running `01_index.py` and `verify_stage1.py`) has been audited against the methodology requirements defined in `slotA-methodology.md`. 

**Verdict:** Everything ran **100% as expected**. All index artifacts were built, normalized, persisted, aligned, and verified via cold reload in a fresh process.

---

## 2. Key Verification Metrics & Evidence

| Component / Test | Expected Benchmark | Measured Result | Status |
|---|---|---|---|
| **Corpus Stats** | 5,183 docs / 300 test queries | 5,183 docs / 300 test queries / 339 qrels pairs / 283 distinct gold docs | **PASS** |
| **Token Audit** | Token limit 512 | 455/5,183 docs (8.78%) exceed 512 tokens. **16.3% of gold docs (46/283)** exceed 512 tokens | **PASS** (Chunking justified) |
| **Chunking Policy** | Auto-select based on >1% overage | Switched to `chunked` (overlap 64). Produced 5,661 units (933 chunks from 455 docs) | **PASS** |
| **FAISS Dense Index** | `BAAI/bge-base-en-v1.5` fp16 | `IndexFlatIP`, shape (5661, 768), float32 normalized L2 (`‖v‖ = 1.000000`) | **PASS** |
| **BM25 Sparse Index** | `rank_bm25.BM25Okapi` doc-level | 5,183 docs, vocab size 87,379 | **PASS** |
| **Cold Reload Test** | Fresh process load | All 4 artifacts (`faiss.index`, `bm25.pkl`, `docmap.json`, `index_meta.json`) reloaded cleanly | **PASS** |
| **Ordinal Round-Trip** | Multi-chunk doc mapping | Deepest doc `10749308` (5 chunks, ordinals 1987–1991) and `9967265` (2 chunks, 1860–1861) confirmed contiguous & bi-directional | **PASS** |
| **Vector Alignment Proof** | Cosine(stored, fresh) ≥ 0.99 | Cosine = **1.000000** for all probed ordinals; Negative control cosine = **0.645611** (< 1.0) | **PASS** |
| **VRAM & Latency** | T4 16GB ceiling | Peak allocated VRAM: 0.82 GiB / reserved 1.07 GiB. Wall-clock embed time: 30.9s | **PASS** |

---

## 3. Important Takeaways & Stage 2 Obligations

1. **Why Chunking Was the Right Tradeoff:**  
   16.3% of gold evidence documents exceed 512 tokens because scientific abstracts have structured sections (RESULTS/CONCLUSIONS at the end). Truncation would have endangered 53 qrels pairs (15.6%). Chunking ensures both BM25 and Dense arms see all text without data loss.

2. **Stage 2 Mandatory Obligations:**
   - **Chunk-to-Doc Collapse:** Retrieval candidate scores across the 5,661 units **must** be collapsed to document level (max score aggregation per docid) before evaluating against doc-level qrels.
   - **RRF Fusion on Collapsed Lists:** Reciprocal Rank Fusion (RRF) must operate on collapsed doc-level rank lists, not unit/chunk rank lists, to prevent long multi-chunk documents from flooding fusion slots.

---

## 4. Environment & Colab Run Instructions

No files need to be modified for Stage 1 as it is already complete and verified. 

### Files Present for Stage 1:
- `01_index.py`
- `verify_stage1.py`
- `src/common.py`

### Commands Used in Colab to Run Stage 1:
```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT
os.chdir(PROJECT)

# Run index build
!python 01_index.py

# Verify index artifacts in fresh process
!python verify_stage1.py
```

Stage 1 is fully verified and ready. Proceeding to Stage 2 implementation.

# Stage 2 — Phase 1 Retrieval & Metric Evaluation Implementation Report

**Status:** IMPLEMENTED & READY FOR COLAB GPU RUN  
**Date:** 2026-08-06  
**Target Stage:** Stage 2 (Phase 1 Baseline Retrieval & Metric Evaluation)

---

## 1. Executive Summary & Cleanup Actions

The project structure has been refactored to eliminate root directory clutter:
- All pipeline and verification scripts were moved into the `src/` directory.
- Root scripts (`01_index.py`, `verify_stage1.py`, `env_check.py`) were removed from the root directory.
- `src/common.py` dynamic path resolution guarantees `PROJECT_ROOT` resolves cleanly whether scripts are executed from root or inside `src/`.

---

## 2. Stage 2 Code Components Implemented

The following files were created in `src/`:

1. **`src/metrics.py`**
   - Pure-Python implementations of **Recall@k**, **MRR**, **nDCG@k** (binary relevance), and **P@k**.
   - Updated default cutoffs `k_list=(1, 5, 10, 20, 50)` to ensure `p@1` is calculated and populated in the metric dictionary (resolving `KeyError: 'p@1'`).
   - Built-in canary unit test function (`test_canary_metrics()`) verifying exact calculation against reference repo B's test vector (`Recall@5 = 0.750`, `MRR = 0.625`, `nDCG@5 = 0.627`).

2. **`src/02_retrieve.py`**
   - **BM25 Arm:** Executes doc-level `rank_bm25` candidate search for top-50 documents per query across all 300 test queries.
   - **Dense Arm:** Encodes test queries with `BAAI/bge-base-en-v1.5` (fp16 on CUDA with SDPA attention), prepending `BGE_QUERY_INSTRUCTION`. Performs FAISS inner-product search and **collapses unit/chunk scores to document level using max score aggregation**.
   - **RRF Hybrid Arm:** Merges doc-level BM25 top-50 and collapsed Dense top-50 candidate rank lists via Reciprocal Rank Fusion ($k=60$).
   - **Latency Timing:** Measures per-query timing (mean & p95 ms/query) for all 3 arms.
   - **Artifact Output:** Persists candidate runs and evaluation table to `artifacts/runs/` (`retrieval_bm25.json`, `retrieval_dense.json`, `retrieval_rrf.json`, `stage2_retrieval_metrics.json`).

3. **`src/verify_stage2.py`**
   - Cold reload verifier running in an independent process.
   - Validates candidate run structures (300 test queries, 50 candidates/query, valid docid resolution).
   - Runs cross-validation parity checks against `pytrec_eval` and `ranx` to confirm 100% numerical match with hand-written metrics.
   - Formats and displays the Phase 1 Baseline Retrieval Performance Table.

---

## 3. Files to Upload to Colab

The entire `src/` directory should be synchronized/uploaded to your Google Drive project root (`/content/drive/MyDrive/slotA-rag-harness/src/`):

- `src/common.py`
- `src/env_check.py`
- `src/01_index.py`
- `src/verify_stage1.py`
- `src/metrics.py`
- `src/02_retrieve.py`
- `src/verify_stage2.py`

---

## 4. Colab Execution Commands

In your Colab notebook, run the following code blocks to execute Stage 2 and verify the Phase 1 retrieval table:

```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT
os.chdir(PROJECT)

# Execute candidate retrieval & metric evaluation across all 3 arms
!python src/02_retrieve.py

# Verify outputs & run independent library parity check in fresh process
!python src/verify_stage2.py
```

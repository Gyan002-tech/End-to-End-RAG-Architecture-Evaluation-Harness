# 3-Tier Dense Model Size Sweet-Spot Evaluation

**Status:** IMPLEMENTED & READY FOR COLAB GPU RUN  
**Date:** 2026-08-07  
**Target Goal:** Provide empirical proof for interview defense that `BAAI/bge-base-en-v1.5` (109M params, 768-dim) is the exact **parameter sweet spot** by comparing it against both a smaller model (`BAAI/bge-small-en-v1.5`, 33M params) and a larger model (`intfloat/e5-large-v2`, 335M params).

---

## 1. The 3-Tier Model Size Comparison Matrix

| Tier | Model Champion | Parameters | Vector Dim | Purpose / Role |
|---|---|---|---|---|
| **Small** | `BAAI/bge-small-en-v1.5` | **33M** (~3x smaller) | **384-dim** | Evaluates if a hyper-light model can achieve sufficient recall |
| **Base ★** | `BAAI/bge-base-en-v1.5` | **109M** (Baseline) | **768-dim** | Our candidate dense champion for Phase 1 |
| **Large** | `intfloat/e5-large-v2` | **335M** (~3x larger) | **1024-dim** | Evaluates if a heavy model justifies its extra latency/memory |

---

## 2. Files Implemented in `src/`

1. **`src/01c_index_bge_small.py`**
   - Embeds 5,183 SciFact documents with `BAAI/bge-small-en-v1.5` (384-dim, fp16 on CUDA).
   - Builds 384-dim FAISS `IndexFlatIP` saved to `artifacts/index/faiss_bge_small.index`.
   - Persists `artifacts/index/docmap_bge_small.json` and `artifacts/index/index_meta_bge_small.json`.

2. **`src/02c_retrieve_bge_small.py`**
   - Encodes 300 test queries with BGE instruction prefix.
   - Searches FAISS index for top-50 candidates per query.
   - Evaluates `Recall@10`, `MRR`, `nDCG@10`, `P@1`, and `Latency (ms)`.
   - Assembles the complete **3-Tier Sweet-Spot Matrix** persisted in `artifacts/runs/dense_model_size_sweetspot.json`.

3. **`src/verify_bge_small.py`**
   - Cold reload verifier checking 384-dim FAISS assertions, 300 test query runs, and displaying the complete 3-tier sweet spot matrix.

---

## 3. Files to Upload to Colab

Upload the new `src/` files to your Google Drive project directory (`/content/drive/MyDrive/slotA-rag-harness/src/`):

- `src/01c_index_bge_small.py`
- `src/02c_retrieve_bge_small.py`
- `src/verify_bge_small.py`

---

## 4. Colab Execution Commands

Run the following commands in Colab to build the `bge-small` index, evaluate core retrieval metrics, and render the complete 3-Tier Model Size Sweet-Spot Table:

```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT
os.chdir(PROJECT)

# 1. Build FAISS index for bge-small-en-v1.5 (384-dim)
!python src/01c_index_bge_small.py

# 2. Evaluate bge-small retrieval & assemble 3-tier sweet spot matrix
!python src/02c_retrieve_bge_small.py

# 3. Verify artifacts & render 3-tier sweet spot table
!python src/verify_bge_small.py
```

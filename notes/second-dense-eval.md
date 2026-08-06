# 2nd Dense Embedding Model Evaluation (`intfloat/e5-large-v2`)

**Status:** IMPLEMENTED & READY FOR COLAB GPU RUN  
**Date:** 2026-08-06  
**Target Goal:** Benchmark a 2nd dense embedding model (`intfloat/e5-large-v2`) against the current champion (`BAAI/bge-base-en-v1.5`) at the core retrieval layer.

---

## 1. Overview & Model Specifications

To evaluate whether a different dense bi-encoder architecture achieves higher initial recall than `bge-base-en-v1.5`, we added `intfloat/e5-large-v2`:

| Model | Embedding Dim | Passage Prefix | Query Prefix | Max Seq Length |
|---|---|---|---|---|
| **`BAAI/bge-base-en-v1.5`** | 768 | (None, raw text) | `"Represent this sentence for searching relevant passages: "` | 512 |
| **`intfloat/e5-large-v2`** | 1024 | `"passage: "` | `"query: "` | 512 |

---

## 2. Files Implemented in `src/`

1. **`src/01b_index_e5.py`**
   - Embeds 5,183 SciFact documents with `intfloat/e5-large-v2` (`passage: ` prefix) in fp16 on CUDA.
   - Builds 1024-dim FAISS `IndexFlatIP` saved to `artifacts/index/faiss_e5.index`.
   - Persists `artifacts/index/docmap_e5.json` and `artifacts/index/index_meta_e5.json`.

2. **`src/02b_retrieve_e5.py`**
   - Encodes 300 test queries with `query: ` prefix.
   - Searches FAISS index for top-50 candidates per query.
   - Evaluates `Recall@10`, `MRR`, `nDCG@10`, `P@1`, and `Latency (ms)`.
   - Generates side-by-side comparison JSON `artifacts/runs/e5_vs_bge_comparison.json`.

3. **`src/verify_e5.py`**
   - Cold reload verifier checking 1024-dim FAISS assertions, 300 test query runs, and displaying the comparative table.

---

## 3. Files to Upload to Colab

Synchronize/upload the new `src/` files to your Google Drive project directory (`/content/drive/MyDrive/slotA-rag-harness/src/`):

- `src/01b_index_e5.py`
- `src/02b_retrieve_e5.py`
- `src/verify_e5.py`

---

## 4. Colab Execution Commands

Run the following commands in Colab to build the `e5-large-v2` index, evaluate core retrieval metrics, and compare against `bge-base-en-v1.5`:

```python
import os
PROJECT = '/content/drive/MyDrive/slotA-rag-harness'
os.environ['SLOTA_ROOT'] = PROJECT
os.chdir(PROJECT)

# Build FAISS index for e5-large-v2
!python src/01b_index_e5.py

# Evaluate e5-large-v2 candidate retrieval & benchmark against bge-base
!python src/02b_retrieve_e5.py

# Verify e5 artifacts & comparative table in fresh process
!python src/verify_e5.py
```

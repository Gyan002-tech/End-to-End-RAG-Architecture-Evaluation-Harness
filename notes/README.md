# Project Documentation & Notes Index

This directory contains architectural specifications, stage execution logs, and benchmark analyses for the **RAG Retrieval + Evaluation Harness**.

---

## Directory Structure

```
notes/
├── specs/       # Stage architecture specifications & design decisions
├── logs/        # Empirical execution logs & verification outputs
└── benchmarks/  # Model size & retrieval performance comparison studies
```

---

## 1. Specifications (`specs/`)

| File | Stage | Description |
| :--- | :---: | :--- |
| [`repo-read.md`](specs/repo-read.md) | Stage 0 | Initial reference repository study & technical analysis |
| [`stage0-plan.md`](specs/stage0-plan.md) | Stage 0 | Hardware verification & setup requirements |
| [`stage1-index.md`](specs/stage1-index.md) | Stage 1 | Indexing methodology, chunking policy, & FAISS/BM25 design |
| [`stage2-retrieval.md`](specs/stage2-retrieval.md) | Stage 2 | Candidate retrieval arms (BM25, Dense, RRF) & IR metrics design |
| [`stage3-rerank.md`](specs/stage3-rerank.md) | Stage 3 | Multi-model staged reranking (BGE-large & Gemma-2-2B) & Pareto grid |
| [`stage4-generation.md`](specs/stage4-generation.md) | Stage 4 | Local RAG answer generation & prompt engineering design |
| [`stage5-judge.md`](specs/stage5-judge.md) | Stage 5 | Local Qwen2.5-7B faithfulness LLM judge design & JSON rubric |

---

## 2. Execution Logs & Outputs (`logs/`)

| Log File | Stage | Description |
| :--- | :---: | :--- |
| [`stage0-output.md`](logs/stage0-output.md) | Stage 0 | GPU hardware & environment validation log |
| [`stage1-output.md`](logs/stage1-output.md) | Stage 1 | FAISS + BM25 index build execution log |
| [`stage1-verification.md`](logs/stage1-verification.md) | Stage 1 | FAISS index cold-reload & cosine alignment verification |
| [`stage2-output.md`](logs/stage2-output.md) | Stage 2 | Candidate retrieval execution log |
| [`stage2-verification.md`](logs/stage2-verification.md) | Stage 2 | IR metrics parity verification output |
| [`stage3-output.md`](logs/stage3-output.md) | Stage 3 | Reranking 3x3 factorial grid execution log |
| [`stage3-verification.md`](logs/stage3-verification.md) | Stage 3 | Pareto frontier survivor verification output |
| [`stage4-output.md`](logs/stage4-output.md) | Stage 4 | RAG answer generation baseline log |
| [`stage4-output-promptUpdate.md`](logs/stage4-output-promptUpdate.md) | Stage 4 | Enhanced guardrail prompt generation log |
| [`stage5-output.md`](logs/stage5-output.md) | Stage 5 | LLM judge baseline evaluation log |
| [`stage5-output-promptUpdate.md`](logs/stage5-output-promptUpdate.md) | Stage 5 | 3-step CoT & entailment judge evaluation log |

---

## 3. Benchmarks (`benchmarks/`)

| File | Description |
| :--- | :--- |
| [`dense-model-benchmarks.md`](benchmarks/dense-model-benchmarks.md) | 3-Tier dense embedding model size sweet-spot study (`bge-small` 33M vs `bge-base` 109M vs `e5-large` 335M) |
| [`e5-vs-bge-comparison.md`](benchmarks/e5-vs-bge-comparison.md) | Detailed baseline comparison of `BAAI/bge-base-en-v1.5` vs `intfloat/e5-large-v2` |

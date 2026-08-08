# Run !python src/03_rerank.py

==========================================================================
[1] Load Stage 2 Candidate Runs & SciFact Document Corpus
==========================================================================
100% 5183/5183 [00:00<00:00, 96105.56it/s]

==========================================================================
[2] Stage 3A: Load bge-reranker-v2-m3 & Rerank Candidates
==========================================================================
  [SKIPPED - CACHED] rerank_m3_bm25.json (nDCG@10: 0.6693, rerank: 1781.12 ms/query)
  [SKIPPED - CACHED] rerank_m3_dense.json (nDCG@10: 0.7420, rerank: 1828.61 ms/query)
  [SKIPPED - CACHED] rerank_m3_rrf.json (nDCG@10: 0.7347, rerank: 1816.43 ms/query)

==========================================================================
[3] Stage 3B: Load bge-reranker-v2-gemma & Rerank Candidates
==========================================================================
  Loading BAAI/bge-reranker-v2-gemma onto CUDA...
2026-08-06 15:18:35.734986: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
tokenizer_config.json: 1.11kB [00:00, 4.90MB/s]
tokenizer.model: 100% 4.24M/4.24M [00:00<00:00, 16.9MB/s]
tokenizer.json: 100% 17.5M/17.5M [00:00<00:00, 54.2MB/s]
special_tokens_map.json: 100% 555/555 [00:00<00:00, 5.45MB/s]
config.json: 100% 659/659 [00:00<00:00, 6.40MB/s]
model.safetensors.index.json: 13.5kB [00:00, 55.0MB/s]
Downloading shards:   0% 0/3 [00:00<?, ?it/s]
Downloading shards: 100% 3/3 [02:45<00:00, 55.04s/it]
`config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
`config.hidden_activation` if you want to override this behaviour.
See https://github.com/huggingface/transformers/pull/29402 for more details.
Loading checkpoint shards: 100% 3/3 [00:01<00:00,  1.78it/s]
generation_config.json: 100% 132/132 [00:00<00:00, 1.43MB/s]

  Reranking BM25 candidates with bge-v2-gemma (batch_size=4)...
pre tokenize: 100% 13/13 [00:00<00:00, 320.84it/s]
You're using a GemmaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
pre tokenize: 100% 13/13 [00:00<00:00, 580.44it/s]
100% 13/13 [00:06<00:00,  2.16it/s]
    Wrote: rerank_gemma_bm25.json (nDCG@10: 0.6956, rerank: 6207.14 ms/query)

  Reranking Dense candidates with bge-v2-gemma (batch_size=4)...
pre tokenize: 100% 13/13 [00:00<00:00, 522.66it/s]
100% 13/13 [00:06<00:00,  2.05it/s]
    Wrote: rerank_gemma_dense.json (nDCG@10: 0.7844, rerank: 6373.62 ms/query)

  Reranking RRF Hybrid candidates with bge-v2-gemma (batch_size=4)...
100% 13/13 [00:06<00:00,  2.14it/s]
    Wrote: rerank_gemma_rrf.json (nDCG@10: 0.7796, rerank: 6233.24 ms/query)

  Unloading bge-v2-gemma model from GPU VRAM...

==========================================================================
[4] Assemble 3x3 Factorial Grid Matrix & Pareto Frontier
==========================================================================

Full 3x3 Factorial Grid Matrix (9 Cells):
Config                       | Recall@10  | MRR      | nDCG@10  | P@1    | Total Latency  | Frontier
-----------------------------------------------------------------------------------------------
Dense -> bge-v2-gemma        | 0.9092     | 0.7508   | 0.7844   | 0.6667 | 6376.81        | ★ SURVIVOR
RRF Hybrid -> bge-v2-gemma   | 0.9059     | 0.7453   | 0.7796   | 0.6633 | 6256.78        | ★ SURVIVOR
Dense -> bge-v2-m3           | 0.8559     | 0.7166   | 0.7420   | 0.6367 | 1831.80        | ★ SURVIVOR
Dense (bge-base) -> none     | 0.8709     | 0.7085   | 0.7407   | 0.6200 | 3.19           | ★ SURVIVOR
RRF Hybrid -> bge-v2-m3      | 0.8512     | 0.7076   | 0.7347   | 0.6267 | 1839.97        | Dominated
BM25 -> bge-v2-gemma         | 0.7668     | 0.6796   | 0.6956   | 0.6200 | 6227.42        | Dominated
RRF Hybrid -> none           | 0.8267     | 0.6395   | 0.6758   | 0.5367 | 23.54          | Dominated
BM25 -> bge-v2-m3            | 0.7462     | 0.6518   | 0.6693   | 0.5900 | 1801.40        | Dominated
BM25 (sparse) -> none        | 0.6862     | 0.5288   | 0.5597   | 0.4367 | 20.28          | Dominated

  Wrote final summary: stage3_rerank_metrics.json

==========================================================================
Stage 3 complete
==========================================================================
  3x3 Factorial Grid computed across 9 cells. Selected 4 Pareto survivors.
  Next: python src/verify_stage3.py


# Output of !python src/verify_stage3.py

==========================================================================
[1] Cold Reload All 9 Run JSON Artifacts from disk
==========================================================================
  Loaded : retrieval_bm25.json          ( 1499.8 KiB)
  Loaded : retrieval_dense.json         ( 1505.0 KiB)
  Loaded : retrieval_rrf.json           ( 1523.3 KiB)
  Loaded : rerank_m3_bm25.json          ( 1539.0 KiB)
  Loaded : rerank_m3_dense.json         ( 1537.3 KiB)
  Loaded : rerank_m3_rrf.json           ( 1536.8 KiB)
  Loaded : rerank_gemma_bm25.json       ( 1516.6 KiB)
  Loaded : rerank_gemma_dense.json      ( 1514.0 KiB)
  Loaded : rerank_gemma_rrf.json        ( 1514.0 KiB)
  Loaded : stage3_rerank_metrics.json   (    4.6 KiB)
100% 5183/5183 [00:00<00:00, 94321.70it/s]

==========================================================================
[2] Validate Candidate Structure & Docid Resolution across 9 Runs
==========================================================================
  bm25                             : queries=300 candidates/q=50 (OK) unresolved=0
  dense                            : queries=300 candidates/q=50 (OK) unresolved=0
  rrf_hybrid                       : queries=300 candidates/q=50 (OK) unresolved=0
  BM25 -> bge-v2-m3                : queries=300 candidates/q=50 (OK) unresolved=0
  Dense -> bge-v2-m3               : queries=300 candidates/q=50 (OK) unresolved=0
  RRF Hybrid -> bge-v2-m3          : queries=300 candidates/q=50 (OK) unresolved=0
  BM25 -> bge-v2-gemma             : queries=300 candidates/q=50 (OK) unresolved=0
  Dense -> bge-v2-gemma            : queries=300 candidates/q=50 (OK) unresolved=0
  RRF Hybrid -> bge-v2-gemma       : queries=300 candidates/q=50 (OK) unresolved=0

==========================================================================
[3] Metric Library Parity Cross-Validation (pytrec_eval)
==========================================================================
  bm25                             pytrec_eval parity : OK
  dense                            pytrec_eval parity : OK
  rrf_hybrid                       pytrec_eval parity : OK
  BM25 -> bge-v2-m3                pytrec_eval parity : OK
  Dense -> bge-v2-m3               pytrec_eval parity : OK
  RRF Hybrid -> bge-v2-m3          pytrec_eval parity : OK
  BM25 -> bge-v2-gemma             pytrec_eval parity : OK
  Dense -> bge-v2-gemma            pytrec_eval parity : OK
  RRF Hybrid -> bge-v2-gemma       pytrec_eval parity : OK

==========================================================================
[4] Render Complete 3x3 Factorial Grid Matrix (9 Cells)
==========================================================================
Config                       | Recall@10  | MRR      | nDCG@10  | P@1    | Total Latency  | Frontier
-----------------------------------------------------------------------------------------------
Dense -> bge-v2-gemma        | 0.9092     | 0.7508   | 0.7844   | 0.6667 | 6376.81        | ★ SURVIVOR
RRF Hybrid -> bge-v2-gemma   | 0.9059     | 0.7453   | 0.7796   | 0.6633 | 6256.78        | ★ SURVIVOR
Dense -> bge-v2-m3           | 0.8559     | 0.7166   | 0.7420   | 0.6367 | 1831.80        | ★ SURVIVOR
Dense (bge-base) -> none     | 0.8709     | 0.7085   | 0.7407   | 0.6200 | 3.19           | ★ SURVIVOR
RRF Hybrid -> bge-v2-m3      | 0.8512     | 0.7076   | 0.7347   | 0.6267 | 1839.97        | Dominated
BM25 -> bge-v2-gemma         | 0.7668     | 0.6796   | 0.6956   | 0.6200 | 6227.42        | Dominated
RRF Hybrid -> none           | 0.8267     | 0.6395   | 0.6758   | 0.5367 | 23.54          | Dominated
BM25 -> bge-v2-m3            | 0.7462     | 0.6518   | 0.6693   | 0.5900 | 1801.40        | Dominated
BM25 (sparse) -> none        | 0.6862     | 0.5288   | 0.5597   | 0.4367 | 20.28          | Dominated

Pareto Frontier Survivors (4 configs selected for Phase 2 Generation):
  -> Dense (bge-base) -> none (nDCG@10: 0.7407, Latency: 3.19 ms/query)
  -> Dense -> bge-v2-m3 (nDCG@10: 0.7420, Latency: 1831.80 ms/query)
  -> RRF Hybrid -> bge-v2-gemma (nDCG@10: 0.7796, Latency: 6256.78 ms/query)
  -> Dense -> bge-v2-gemma (nDCG@10: 0.7844, Latency: 6376.81 ms/query)

==========================================================================
VERDICT
==========================================================================
  All Stage 3 checks passed. 9-cell factorial grid matrix & Pareto survivors verified.
  Phase 1 Complete! Phase 2 (Generation & Faithfulness) is unblocked.

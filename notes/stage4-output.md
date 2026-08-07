# Output of running !python src/04_generate.py

==========================================================================
[1] Load SciFact Dataset & Document Corpus
==========================================================================
100% 5183/5183 [00:00<00:00, 97637.46it/s]

==========================================================================
[2] Load Qwen/Qwen2.5-1.5B-Instruct Generator onto CUDA
==========================================================================
  Loading Qwen/Qwen2.5-1.5B-Instruct...
tokenizer_config.json: 7.30kB [00:00, 37.5MB/s]
vocab.json: 2.78MB [00:00, 65.9MB/s]
merges.txt: 1.67MB [00:00, 140MB/s]
tokenizer.json: 7.03MB [00:00, 182MB/s]
config.json: 100% 660/660 [00:00<00:00, 5.10MB/s]
2026-08-07 12:21:52.340141: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
model.safetensors: 100% 3.09G/3.09G [01:04<00:00, 47.7MB/s]
generation_config.json: 100% 242/242 [00:00<00:00, 1.88MB/s]

==========================================================================
[3] Execute Local Answer Generation across Survivor Runs
==========================================================================

  Generating answers for: Dense -> none...
  Generating [Dense -> none]: 100% 300/300 [24:47<00:00,  4.96s/query, mean_lat=4959ms]
    Wrote: gen_dense_none.json      (gen latency: 4958.59 ms/query, truncations: 9)

  Generating answers for: Dense -> bge-v2-m3...
  Generating [Dense -> bge-v2-m3]: 100% 300/300 [24:49<00:00,  4.96s/query, mean_lat=4963ms]
    Wrote: gen_dense_m3.json        (gen latency: 4963.17 ms/query, truncations: 2)

  Generating answers for: RRF Hybrid -> bge-v2-gemma...
  Generating [RRF Hybrid -> bge-v2-gemma]: 100% 300/300 [25:07<00:00,  5.03s/query, mean_lat=5026ms]
    Wrote: gen_rrf_gemma.json       (gen latency: 5025.70 ms/query, truncations: 6)

  Generating answers for: Dense -> bge-v2-gemma...
  Generating [Dense -> bge-v2-gemma]: 100% 300/300 [24:54<00:00,  4.98s/query, mean_lat=4982ms]
    Wrote: gen_dense_gemma.json     (gen latency: 4982.45 ms/query, truncations: 10)

  Unloading Qwen2.5-1.5B-Instruct from GPU VRAM...

==========================================================================
[4] Persist Stage 4 Generation Summary
==========================================================================

Stage 4 Generation Summary:
Config                       | Output File              | Gen Latency (ms)   | Truncations
--------------------------------------------------------------------------------
Dense -> none                | gen_dense_none.json      | 4958.59            | 9
Dense -> bge-v2-m3           | gen_dense_m3.json        | 4963.17            | 2
RRF Hybrid -> bge-v2-gemma   | gen_rrf_gemma.json       | 5025.70            | 6
Dense -> bge-v2-gemma        | gen_dense_gemma.json     | 4982.45            | 10

==========================================================================
Stage 4 complete
==========================================================================
  Generated RAG answers for 4 Pareto survivors. Persisted: stage4_generation_summary.json
  Next: python src/verify_stage4.py


# Output of running !python src/verify_stage4.py

==========================================================================
[1] Cold Reload All 4 Survivor Generation Artifacts from disk
==========================================================================
  Loaded : gen_dense_none.json            (  291.9 KiB)
  Loaded : gen_dense_m3.json              (  291.4 KiB)
  Loaded : gen_rrf_gemma.json             (  295.2 KiB)
  Loaded : gen_dense_gemma.json           (  292.3 KiB)
  Loaded : stage4_generation_summary.json (    0.9 KiB)
100% 5183/5183 [00:00<00:00, 75866.46it/s]

==========================================================================
[2] Validate Answer Completeness & Citation Integrity across Survivor Runs
==========================================================================
  Dense -> none                : answers=300 empty=0 (OK) cited=2.3% mean_len=88.7 words truncs=9
  Dense -> bge-v2-m3           : answers=300 empty=0 (OK) cited=0.3% mean_len=88.7 words truncs=2
  RRF Hybrid -> bge-v2-gemma   : answers=300 empty=0 (OK) cited=1.7% mean_len=90.0 words truncs=6
  Dense -> bge-v2-gemma        : answers=300 empty=0 (OK) cited=2.3% mean_len=88.6 words truncs=10

==========================================================================
[3] Render Stage 4 Generation Latency & Summary Table
==========================================================================
Config                       | Output File              | Gen Latency (ms)   | Truncations
--------------------------------------------------------------------------------
Dense -> none                | gen_dense_none.json      | 4958.59            | 9
Dense -> bge-v2-m3           | gen_dense_m3.json        | 4963.17            | 2
RRF Hybrid -> bge-v2-gemma   | gen_rrf_gemma.json       | 5025.70            | 6
Dense -> bge-v2-gemma        | gen_dense_gemma.json     | 4982.45            | 10

==========================================================================
VERDICT
==========================================================================
  All Stage 4 answer generation checks passed. 1,200 RAG answers verified across 4 survivors.
  Stage 4 Complete! Proceeding to Stage 5 (Local Qwen2.5-7B Faithfulness Judge).

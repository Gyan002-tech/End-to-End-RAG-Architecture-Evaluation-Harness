# Output of running !python src/04_generate.py

==========================================================================
[1] Load SciFact Dataset & Document Corpus
==========================================================================
100% 5183/5183 [00:00<00:00, 55079.42it/s]

==========================================================================
[2] Load Qwen/Qwen2.5-1.5B-Instruct Generator onto CUDA
==========================================================================
  Loading Qwen/Qwen2.5-1.5B-Instruct...
tokenizer_config.json: 7.30kB [00:00, 36.6MB/s]
vocab.json: 2.78MB [00:00, 69.9MB/s]
merges.txt: 1.67MB [00:00, 132MB/s]
tokenizer.json: 7.03MB [00:00, 179MB/s]
config.json: 100% 660/660 [00:00<00:00, 5.77MB/s]
2026-08-07 23:24:11.958010: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
model.safetensors: 100% 3.09G/3.09G [01:03<00:00, 48.9MB/s]
generation_config.json: 100% 242/242 [00:00<00:00, 2.66MB/s]

==========================================================================
[3] Execute Local Answer Generation across Survivor Runs (Per-Query Auto-Save)
==========================================================================

  Generating answers for: Dense -> none...
  Generating [Dense -> none]: 100% 300/300 [26:28<00:00,  5.29s/query, mean_lat=5283ms]

  Generating answers for: Dense -> bge-v2-m3...
  Generating [Dense -> bge-v2-m3]: 100% 300/300 [26:12<00:00,  5.24s/query, mean_lat=5232ms]

  Generating answers for: RRF Hybrid -> bge-v2-gemma...
  Generating [RRF Hybrid -> bge-v2-gemma]: 100% 300/300 [26:23<00:00,  5.28s/query, mean_lat=5266ms]

  Generating answers for: Dense -> bge-v2-gemma...
  Generating [Dense -> bge-v2-gemma]: 100% 300/300 [26:13<00:00,  5.24s/query, mean_lat=5233ms]

  Unloading Qwen2.5-1.5B-Instruct from GPU VRAM...

==========================================================================
[4] Persist Stage 4 Generation Summary
==========================================================================

Stage 4 Generation Summary:
Config                       | Output File              | Gen Latency (ms)   | Truncations
--------------------------------------------------------------------------------
Dense -> none                | gen_dense_none.json      | 5283.43            | 4
Dense -> bge-v2-m3           | gen_dense_m3.json        | 5232.06            | 5
RRF Hybrid -> bge-v2-gemma   | gen_rrf_gemma.json       | 5266.41            | 5
Dense -> bge-v2-gemma        | gen_dense_gemma.json     | 5233.41            | 5

==========================================================================
Stage 4 complete
==========================================================================
  Generated RAG answers for 4 Pareto survivors. Persisted: stage4_generation_summary.json
  Next: python src/verify_stage4.py


# Output of running !python src/verify_stage4.py


==========================================================================
[1] Cold Reload All 4 Survivor Generation Artifacts from disk
==========================================================================
  Loaded : gen_dense_none.json            (  301.8 KiB)
  Loaded : gen_dense_m3.json              (  297.9 KiB)
  Loaded : gen_rrf_gemma.json             (  302.2 KiB)
  Loaded : gen_dense_gemma.json           (  298.8 KiB)
  Loaded : stage4_generation_summary.json (    0.9 KiB)
100% 5183/5183 [00:00<00:00, 86297.34it/s]

==========================================================================
[2] Validate Answer Completeness & Citation Integrity across Survivor Runs
==========================================================================
  Dense -> none                : answers=300 empty=0 (OK) cited=1.3% mean_len=91.0 words truncs=4
  Dense -> bge-v2-m3           : answers=300 empty=0 (OK) cited=2.7% mean_len=89.0 words truncs=5
  RRF Hybrid -> bge-v2-gemma   : answers=300 empty=0 (OK) cited=3.3% mean_len=90.8 words truncs=5
  Dense -> bge-v2-gemma        : answers=300 empty=0 (OK) cited=3.3% mean_len=89.5 words truncs=5

==========================================================================
[3] Render Stage 4 Generation Latency & Summary Table
==========================================================================
Config                       | Output File              | Gen Latency (ms)   | Truncations
--------------------------------------------------------------------------------
Dense -> none                | gen_dense_none.json      | 5283.43            | 4
Dense -> bge-v2-m3           | gen_dense_m3.json        | 5232.06            | 5
RRF Hybrid -> bge-v2-gemma   | gen_rrf_gemma.json       | 5266.41            | 5
Dense -> bge-v2-gemma        | gen_dense_gemma.json     | 5233.41            | 5

==========================================================================
VERDICT
==========================================================================
  All Stage 4 answer generation checks passed. 1,200 RAG answers verified across 4 survivors.
  Stage 4 Complete! Proceeding to Stage 5 (Local Qwen2.5-7B Faithfulness Judge).

# nvidia-smi result
Tue Aug  4 19:20:12 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   48C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# env_check.py Output
========================================================================
Slot A — Stage 0 environment check
========================================================================

[0] Host
  python............................ 3.12.13
  platform.......................... Linux-6.6.122+-x86_64-with-glibc2.35

[1] torch / CUDA
  torch.__version__................. 2.11.0+cu128
  torch.version.cuda................ 12.8
  cuDNN............................. 91900
  torch.cuda.is_available()......... True
  torch.cuda.device_count()......... 1

[2] GPU device
  device name....................... Tesla T4
  compute capability................ sm_75  (tuple (7, 5))
  total VRAM........................ 14.56 GiB
  multiprocessors (SMs)............. 40
  arch list in this build........... sm_75, sm_80, sm_86, sm_90, sm_100, sm_120

[3] Precision support (plan requires fp16, forbids bf16)
  native bf16 (cc >= 8.0)?.......... False
  torch.cuda.is_bf16_supported().... True   <- may count emulation, do not trust
  empirical fp16 matmul............. OK
  empirical bf16 matmul............. OK

[4] Attention backend (plan requires SDPA, forbids FlashAttention-2)
  F.scaled_dot_product_attention.... present
  flash_attn installed?............. no (correct; ModuleNotFoundError)
     -> pass attn_implementation="sdpa" to every from_pretrained() call.

[5] Installed versions
  torch............................. 2.11.0+cu128  (dist: torch)
  transformers...................... 4.46.3  (dist: transformers)
  faiss............................. 1.9.0  (dist: faiss-cpu)
  rank_bm25......................... 0.2.2  (dist: rank-bm25)
  ranx.............................. 0.3.20  (dist: ranx)
  pytrec_eval....................... 0.5.10  (dist: pytrec-eval-terrier)
  numpy............................. 1.26.4  (dist: numpy)
  sentence-transformers............. 3.3.1  (dist: sentence-transformers)
  FlagEmbedding..................... 1.3.5  (dist: FlagEmbedding)
  beir.............................. 2.0.0  (dist: beir)
  bitsandbytes...................... 0.44.1  (dist: bitsandbytes)
  accelerate........................ 1.1.1  (dist: accelerate)
  numba............................. 0.60.0  (dist: numba)

[6] Import smoke (a version string does not prove the module loads)
  import faiss...................... OK
  import rank_bm25.................. OK
  import ranx....................... OK
  import pytrec_eval................ OK
2026-08-04 19:46:21.393488: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  import sentence_transformers...... OK
  import FlagEmbedding.............. OK
  import beir....................... OK
  faiss.__version__................. 1.9.0
  faiss GPUs visible................ 0  (0 expected — plan uses CPU IndexFlatIP)

========================================================================
VERDICT
========================================================================
  [OK] T4 / sm_75 / fp16 / SDPA — matches slotA-methodology.md §3.

#!/usr/bin/env python3
"""Stage 0 gate: verify the runtime matches what slotA-methodology.md §3 assumes.

Usage:
    python src/env_check.py              # exits 1 if no CUDA GPU is visible
    python src/env_check.py --allow-cpu  # report only, always exit 0 (CPU stages 0-2)
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_GPU_SUBSTRING = "T4"
EXPECTED_CAPABILITY = (7, 5)  # sm_75, Turing
BF16_MIN_CAPABILITY = (8, 0)  # native bf16 needs Ampere+


def _line(label: str, value: object) -> None:
    print(f"  {label:.<34} {value}")


def _dist_version(*candidates: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    for name in candidates:
        try:
            return f"{version(name)}  (dist: {name})"
        except PackageNotFoundError:
            continue
    return "NOT INSTALLED"


def _importable(module: str) -> tuple[bool, str]:
    import importlib

    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def check_torch_and_gpu() -> dict:
    findings: dict = {"cuda": False, "device_name": None, "capability": None}

    print("\n[1] torch / CUDA")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _line("torch", f"IMPORT FAILED — {type(exc).__name__}: {exc}")
        return findings

    _line("torch.__version__", torch.__version__)
    _line("torch.version.cuda", torch.version.cuda)
    _line("cuDNN", getattr(torch.backends.cudnn, "version", lambda: None)())

    cuda_available = torch.cuda.is_available()
    findings["cuda"] = cuda_available
    _line("torch.cuda.is_available()", cuda_available)

    if not cuda_available:
        print("\n  !! No CUDA device visible.")
        print("     In Colab: Runtime -> Change runtime type -> Hardware accelerator: GPU (T4).")
        return findings

    _line("torch.cuda.device_count()", torch.cuda.device_count())

    print("\n[2] GPU device")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    findings["device_name"] = name
    findings["capability"] = cap

    _line("device name", name)
    _line("compute capability", f"sm_{cap[0]}{cap[1]}  (tuple {cap})")
    _line("total VRAM", f"{props.total_memory / 1024**3:.2f} GiB")
    _line("multiprocessors (SMs)", props.multi_processor_count)
    _line("arch list in this build", ", ".join(torch.cuda.get_arch_list()))

    print("\n[3] Precision support (plan requires fp16, forbids bf16)")
    native_bf16 = cap >= BF16_MIN_CAPABILITY
    _line("native bf16 (cc >= 8.0)?", native_bf16)

    try:
        torch_says = torch.cuda.is_bf16_supported()
    except Exception as exc:  # noqa: BLE001
        torch_says = f"raised {type(exc).__name__}"
    _line("torch.cuda.is_bf16_supported()", f"{torch_says}   <- may count emulation, do not trust")

    for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        try:
            a = torch.randn(64, 64, device="cuda", dtype=dtype)
            (a @ a).sum().item()
            torch.cuda.synchronize()
            _line(f"empirical {label} matmul", "OK")
        except Exception as exc:  # noqa: BLE001
            _line(f"empirical {label} matmul", f"FAILED — {type(exc).__name__}: {exc}")

    print("\n[4] Attention backend (plan requires SDPA, forbids FlashAttention-2)")
    has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    _line("F.scaled_dot_product_attention", "present" if has_sdpa else "MISSING")
    fa_ok, fa_err = _importable("flash_attn")
    _line(
        "flash_attn installed?",
        "yes — IGNORE IT, needs sm_80+" if fa_ok else f"no (correct; {fa_err.split(':')[0]})",
    )
    print('     -> pass attn_implementation="sdpa" to every from_pretrained() call.')

    return findings


def check_versions() -> None:
    print("\n[5] Installed versions")
    _line("torch", _dist_version("torch"))
    _line("transformers", _dist_version("transformers"))
    _line("faiss", _dist_version("faiss-cpu", "faiss", "faiss-gpu"))
    _line("rank_bm25", _dist_version("rank-bm25", "rank_bm25"))
    _line("ranx", _dist_version("ranx"))
    _line("pytrec_eval", _dist_version("pytrec-eval-terrier", "pytrec_eval", "pytrec-eval"))
    _line("numpy", _dist_version("numpy"))
    _line("sentence-transformers", _dist_version("sentence-transformers"))
    _line("FlagEmbedding", _dist_version("FlagEmbedding"))
    _line("beir", _dist_version("beir"))
    _line("bitsandbytes", _dist_version("bitsandbytes"))
    _line("accelerate", _dist_version("accelerate"))
    _line("numba", _dist_version("numba"))

    print("\n[6] Import smoke (a version string does not prove the module loads)")
    for module in (
        "faiss",
        "rank_bm25",
        "ranx",
        "pytrec_eval",
        "sentence_transformers",
        "FlagEmbedding",
        "beir",
    ):
        ok, err = _importable(module)
        _line(f"import {module}", "OK" if ok else f"FAILED — {err}")

    try:
        import faiss

        _line("faiss.__version__", getattr(faiss, "__version__", "unknown"))
        n_gpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        _line("faiss GPUs visible", f"{n_gpu}  (0 expected — plan uses CPU IndexFlatIP)")
    except Exception:  # noqa: BLE001
        pass


def verdict(findings: dict, allow_cpu: bool) -> int:
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    problems: list[str] = []

    if not findings["cuda"]:
        problems.append("No CUDA GPU visible — stages 03/04/05 cannot run.")
    else:
        name = findings["device_name"] or ""
        cap = findings["capability"]
        if EXPECTED_GPU_SUBSTRING not in name:
            problems.append(
                f"GPU is {name!r}, NOT a {EXPECTED_GPU_SUBSTRING}. The plan's VRAM budget, "
                "fp16-only rule and no-FlashAttention rule were written for a T4 — "
                "re-check them against this device before trusting any stage."
            )
        if cap != EXPECTED_CAPABILITY:
            problems.append(
                f"Compute capability is sm_{cap[0]}{cap[1]}, plan assumes sm_75. "
                "If cc >= 8.0 you MAY use bf16 and FlashAttention-2 — but that is a "
                "changed variable and must be stated in any bullet."
            )

    if problems:
        for p in problems:
            print(f"  [FLAG] {p}")
    else:
        print("  [OK] T4 / sm_75 / fp16 / SDPA — matches slotA-methodology.md §3.")

    hard_fail = not findings["cuda"] and not allow_cpu
    if hard_fail:
        print("\n  Exiting 1. Re-run with --allow-cpu to proceed on CPU-only stages (0-2).")
        return 1
    if not findings["cuda"]:
        print("\n  --allow-cpu set: CPU-only stages (01_index, 02_retrieve) are still viable.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="report only; do not fail when no CUDA GPU is present",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Slot A — Stage 0 environment check")
    print("=" * 72)
    print("\n[0] Host")
    _line("python", sys.version.split()[0])
    _line("platform", platform.platform())

    findings = check_torch_and_gpu()
    check_versions()
    return verdict(findings, args.allow_cpu)


if __name__ == "__main__":
    raise SystemExit(main())

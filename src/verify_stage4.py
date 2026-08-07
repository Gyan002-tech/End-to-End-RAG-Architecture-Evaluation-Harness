#!/usr/bin/env python3
"""Stage 4 Verification — Cold reload, answer audit, citation & truncation verification.

Runs in a fresh process, loading all 4 generated answer JSONs and summary from disk.

Checks:
  [1] Cold reload all 4 survivor generation JSONs + stage4_generation_summary.json
  [2] Validate answer completeness (300 queries per survivor, non-empty answer text)
  [3] Citation integrity audit (presence of inline '[Document ...]' citations)
  [4] Truncation audit (queries reaching max_new_tokens without EOS)
  [5] Display Generation Latency & Word Count Summary Table

Usage:
    python src/verify_stage4.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import RUNS_DIR, SCIFACT_SPLIT, load_scifact

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    failures: list[str] = []

    hdr("[1] Cold Reload All 4 Survivor Generation Artifacts from disk")
    expected_files = [
        "gen_dense_none.json",
        "gen_dense_m3.json",
        "gen_rrf_gemma.json",
        "gen_dense_gemma.json",
        "stage4_generation_summary.json",
    ]

    loaded_gens: Dict[str, dict] = {}
    for filename in expected_files:
        p = RUNS_DIR / filename
        if not p.exists():
            print(f"  MISSING: {p.name}")
            failures.append(f"Missing generation artifact {p.name}")
        else:
            print(f"  Loaded : {p.name:<30} ({p.stat().st_size / 1024:7.1f} KiB)")
            if filename != "stage4_generation_summary.json":
                with open(p) as f:
                    loaded_gens[filename] = json.load(f)

    if failures:
        return 1

    with open(RUNS_DIR / "stage4_generation_summary.json") as f:
        summary_data = json.load(f)

    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)
    citation_regex = re.compile(r"\[Document\s+\w+\]", re.IGNORECASE)

    hdr("[2] Validate Answer Completeness & Citation Integrity across Survivor Runs")
    for filename, data in loaded_gens.items():
        config_name = data.get("config", filename)
        gens = data["generations"]

        if len(gens) != len(queries):
            failures.append(f"{config_name} has {len(gens)} answers, expected {len(queries)}")

        empty_answers = 0
        cited_answers = 0
        truncated_answers = 0
        word_counts = []

        for qid, entry in gens.items():
            ans = entry.get("answer", "")
            if not ans:
                empty_answers += 1
            else:
                word_counts.append(len(ans.split()))

            if citation_regex.search(ans):
                cited_answers += 1

            if entry.get("hit_max_tokens"):
                truncated_answers += 1

        mean_words = sum(word_counts) / max(1, len(word_counts))
        cite_pct = (cited_answers / max(1, len(gens))) * 100.0

        status_str = "OK" if empty_answers == 0 else "FAIL"
        print(
            f"  {config_name:<28} : answers={len(gens)} empty={empty_answers} ({status_str}) "
            f"cited={cite_pct:.1f}% mean_len={mean_words:.1f} words truncs={truncated_answers}"
        )

        if empty_answers > 0:
            failures.append(f"{config_name} contains empty answer strings")

    hdr("[3] Render Stage 4 Generation Latency & Summary Table")
    survivor_rows = summary_data.get("survivor_generations", [])
    print(f"{'Config':<28} | {'Output File':<24} | {'Gen Latency (ms)':<18} | {'Truncations'}")
    print("-" * 80)
    for r in survivor_rows:
        print(f"{r['config']:<28} | {r['output_file']:<24} | {r['mean_gen_latency_ms']:<18.2f} | {r['truncations_count']}")

    hdr("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All Stage 4 answer generation checks passed. 1,200 RAG answers verified across 4 survivors.")
    print("  Stage 4 Complete! Proceeding to Stage 5 (Local Qwen2.5-7B Faithfulness Judge).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

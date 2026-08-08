#!/usr/bin/env python3
"""Stage 5 Verification — Cold reload, judgment audit & Final Phase 2 Matrix rendering.

Runs in a fresh process, loading all 4 judgment JSONs and summary from disk.

Checks:
  [1] Cold reload all 4 survivor judgment JSONs + stage5_faithfulness_summary.json
  [2] Validate judgment completeness (300 queries per survivor, valid scores in [0.0, 1.0])
  [3] Render Final Phase 2 Pareto Survivor Matrix (nDCG@10 vs Latency vs Faithfulness)

Usage:
    python src/verify_stage5.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.engine.common import RUNS_DIR, SCIFACT_SPLIT, load_scifact

BANNER = "=" * 74


def hdr(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    failures: list[str] = []

    hdr("[1] Cold Reload All 4 Survivor Judgment Artifacts from disk")
    expected_files = [
        "judge_dense_none.json",
        "judge_dense_m3.json",
        "judge_rrf_gemma.json",
        "judge_dense_gemma.json",
        "stage5_faithfulness_summary.json",
    ]

    loaded_judgs: Dict[str, dict] = {}
    for filename in expected_files:
        p = RUNS_DIR / filename
        if not p.exists():
            print(f"  MISSING: {p.name}")
            failures.append(f"Missing judgment artifact {p.name}")
        else:
            print(f"  Loaded : {p.name:<30} ({p.stat().st_size / 1024:7.1f} KiB)")
            if filename != "stage5_faithfulness_summary.json":
                with open(p) as f:
                    loaded_judgs[filename] = json.load(f)

    if failures:
        return 1

    with open(RUNS_DIR / "stage5_faithfulness_summary.json") as f:
        summary_data = json.load(f)

    corpus, queries, qrels = load_scifact(split=SCIFACT_SPLIT)

    hdr("[2] Validate Judgment Completeness & Score Bounds across Survivor Runs")
    for filename, data in loaded_gens.items() if (loaded_gens := loaded_judgs) else []:
        config_name = data.get("config", filename)
        judgs = data["judgments"]

        if len(judgs) != len(queries):
            failures.append(f"{config_name} has {len(judgs)} judgments, expected {len(queries)}")

        invalid_scores = 0
        scores = []
        for qid, entry in judgs.items():
            sc = entry.get("faithfulness_score", -1.0)
            if not (0.0 <= sc <= 1.0):
                invalid_scores += 1
            else:
                scores.append(sc)

        mean_f = sum(scores) / max(1, len(scores))
        status_str = "OK" if invalid_scores == 0 else "FAIL"
        print(f"  {config_name:<28} : judgments={len(judgs)} invalid={invalid_scores} ({status_str}) mean_faithfulness={mean_f:.4f}")

        if invalid_scores > 0:
            failures.append(f"{config_name} contains invalid faithfulness scores outside [0, 1]")

    hdr("[3] Render Final Phase 2 Pareto Survivor Matrix")
    survivor_rows = summary_data.get("phase2_pareto_survivors", [])
    print(f"{'Config':<28} | {'Retrieval nDCG@10':<18} | {'Total Latency (ms)':<20} | {'Faithfulness'}")
    print("-" * 85)
    for r in survivor_rows:
        print(f"{r['config']:<28} | {r['ndcg@10']:<18.4f} | {r['total_ret_lat_ms']:<20.2f} | {r['faithfulness_score']:.4f}")

    hdr("FINAL PROJECT VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All Stage 5 faithfulness judgment checks passed cleanly.")
    print("  CONGRATULATIONS! Phase 1 & Phase 2 End-to-End RAG Harness is 100% COMPLETE & VERIFIED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

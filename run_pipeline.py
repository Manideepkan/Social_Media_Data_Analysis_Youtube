"""
Master Pipeline Runner — YouTube NLP Pipeline
Executes all 6 processing stages sequentially, then optionally launches the dashboard.

Usage:
    python run_pipeline.py            # runs stages 1-6
    python run_pipeline.py --dash     # runs stages 1-6, then launches dashboard
    python run_pipeline.py --from 3   # resume from stage 3
    python run_pipeline.py --only 2   # run only stage 2
"""

import argparse
import importlib
import io
import sys
import time

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

STAGES = [
    (1, "yt_stage1_load",         "Load & Feature Engineering"),
    (2, "yt_stage2_eda",          "EDA & Visualization"),
    (3, "yt_stage3_preprocess",   "Multilingual Preprocessing"),
    (4, "yt_stage4_embeddings",   "Embedding Generation"),
    (5, "yt_stage5_reduction",    "Dimensionality Reduction"),
    (6, "yt_stage6_classification","ML Classification"),
]


def run_stage(num: int, module_name: str, description: str) -> bool:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RUNNING STAGE {num}: {description}")
    print(f"{sep}")
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.perf_counter() - t0
        print(f"\n  [Stage {num} completed in {elapsed:.1f}s]")
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  [Stage {num} FAILED after {elapsed:.1f}s]")
        print(f"  Error: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="YouTube NLP Pipeline Runner")
    parser.add_argument("--dash",   action="store_true", help="Launch dashboard after pipeline")
    parser.add_argument("--from",   dest="from_stage", type=int, default=1,
                        help="Start from this stage (default: 1)")
    parser.add_argument("--only",   dest="only_stage", type=int, default=None,
                        help="Run only this stage")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOUTUBE NLP PIPELINE")
    print("=" * 60)

    wall_start = time.perf_counter()
    stages_to_run = STAGES

    if args.only_stage:
        stages_to_run = [s for s in STAGES if s[0] == args.only_stage]
        if not stages_to_run:
            print(f"  Stage {args.only_stage} not found. Valid: 1-6")
            sys.exit(1)
    else:
        stages_to_run = [s for s in STAGES if s[0] >= args.from_stage]

    print(f"\n  Stages to run: {[s[0] for s in stages_to_run]}")
    print(f"  Total stages : {len(stages_to_run)}")

    results = []
    for num, module, desc in stages_to_run:
        ok = run_stage(num, module, desc)
        results.append((num, desc, ok))
        if not ok:
            print(f"\n  Pipeline halted at stage {num}.")
            break

    wall_elapsed = time.perf_counter() - wall_start

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for num, desc, ok in results:
        status = "OK" if ok else "FAILED"
        mark   = "[✓]" if ok else "[✗]"
        print(f"  {mark} Stage {num}: {desc:<30}  {status}")

    print(f"\n  Total wall time: {wall_elapsed:.1f}s ({wall_elapsed/60:.1f} min)")

    all_ok = all(ok for _, _, ok in results)
    if all_ok:
        print("\n  All stages completed successfully.")
    else:
        print("\n  Pipeline finished with errors. Check output above.")

    # ── Dashboard ─────────────────────────────────────────────────
    if args.dash and all_ok:
        print("\n" + "=" * 60)
        print("  LAUNCHING DASHBOARD (Stage 7)")
        print("=" * 60)
        try:
            import yt_stage7_dashboard
            yt_stage7_dashboard.main()
        except Exception as exc:
            print(f"  Dashboard failed: {exc}")
            import traceback
            traceback.print_exc()
    elif args.dash and not all_ok:
        print("\n  Skipping dashboard — pipeline had errors.")


if __name__ == "__main__":
    main()

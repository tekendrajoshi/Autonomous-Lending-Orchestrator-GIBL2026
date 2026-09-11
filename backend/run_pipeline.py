"""
run_pipeline.py
===============
Command-line interface for the Autonomous Credit Orchestrator.

Usage examples:
    # Process a single applicant
    python run_pipeline.py --mode single --applicant AP-050234 --data-dir "./Track A/structured"

    # Process first 100 applications from CSV files
    python run_pipeline.py --mode batch --limit 100 --data-dir "./Track A/structured"

    # Show a summary of the results
    python run_pipeline.py --mode summary --results-file results/batch_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from orchestrator.data_loader import load_all_tables, get_applicant_state, get_all_applicant_ids
from orchestrator.pipeline import run_single_application, run_batch


def print_result(result: dict):
    """Pretty-print a single result."""
    print("\n" + "=" * 70)
    print(f"APPLICATION  : {result.get('application_id')}")
    print(f"APPLICANT    : {result.get('applicant_id')}")
    print("-" * 70)

    decision = result.get("final_decision", "unknown").upper()
    decision_emoji = {
        "APPROVE": "✅", "CONDITIONAL": "⚠️",
        "REFER": "🔀", "REJECT": "❌", "ERROR": "💥"
    }.get(decision, "❓")

    print(f"DECISION     : {decision_emoji}  {decision}")
    if result.get("approved_amount_nrs"):
        print(f"APPROVED AMT : NRs {result['approved_amount_nrs']:,.0f}")
    if result.get("interest_rate_pct"):
        print(f"INTEREST RATE: {result['interest_rate_pct']}% ({result.get('interest_tier', 'N/A')})")
    print(f"CREDIT SCORE : {result.get('credit_score')} ({result.get('score_band')})")
    print(f"INCOME EST.  : NRs {result.get('income_estimate_monthly', 0):,.0f}/mo "
          f"(confidence: {result.get('income_confidence', 0):.2f})")
    print(f"COMPLIANCE   : {result.get('compliance_status')} | "
          f"Flags: {result.get('compliance_flags', [])}")
    print(f"TIME         : {result.get('processing_time_seconds', 0):.3f}s")
    print(f"\nRATIONALE: {result.get('decision_rationale', 'N/A')}")
    if result.get("error_message"):
        print(f"\n⚠ ERROR: {result['error_message']}")
    print("=" * 70)


def print_batch_summary(results: list[dict]):
    """Print a summary table for a batch run."""
    from collections import Counter

    decisions = [r.get("final_decision", "error") for r in results]
    counts = Counter(decisions)
    total = len(results)

    print("\n" + "=" * 70)
    print(f"BATCH RESULTS SUMMARY — {total} applications processed")
    print("=" * 70)
    for decision, count in sorted(counts.items()):
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {decision.upper():<12}: {count:5,}  ({pct:.1f}%)  {bar}")
    print("-" * 70)

    scores = [r.get("credit_score") for r in results if r.get("credit_score")]
    if scores:
        print(f"  Credit Score : avg={sum(scores)/len(scores):.0f}  "
              f"min={min(scores)}  max={max(scores)}")

    incomes = [r.get("income_estimate_monthly") for r in results if r.get("income_estimate_monthly")]
    if incomes:
        print(f"  Monthly Income: avg=NRs {sum(incomes)/len(incomes):,.0f}  "
              f"min=NRs {min(incomes):,.0f}  max=NRs {max(incomes):,.0f}")

    approved_amts = [r.get("approved_amount_nrs") for r in results if r.get("approved_amount_nrs")]
    if approved_amts:
        print(f"  Approved Amt : avg=NRs {sum(approved_amts)/len(approved_amts):,.0f}  "
              f"total=NRs {sum(approved_amts):,.0f}")

    errors = [r for r in results if r.get("error_message")]
    if errors:
        print(f"\n  ⚠ {len(errors)} application(s) had errors")

    print("=" * 70)


def mode_single(args):
    print(f"[CLI] Loading data from: {args.data_dir}")
    tables = load_all_tables(args.data_dir)

    # Find application_id
    application_id = args.application_id
    if not application_id:
        loans = tables.get("loans")
        if loans is not None:
            mask = loans["applicant_id"].astype(str) == str(args.applicant)
            matches = loans[mask]
            if not matches.empty:
                application_id = str(matches.iloc[0]["application_id"])
    if not application_id:
        application_id = f"LA-MANUAL-{args.applicant}"

    initial_state = get_applicant_state(args.applicant, application_id, tables)
    result = run_single_application(initial_state)
    print_result(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[CLI] Full result saved to {args.output}")


def mode_batch(args):
    print(f"[CLI] Loading data from: {args.data_dir}")
    tables = load_all_tables(args.data_dir)

    pairs = get_all_applicant_ids(tables)
    if not pairs:
        print("[CLI] No loan applications found in the data.")
        sys.exit(1)

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"[CLI] Processing {len(pairs)} applications...")

    states = [get_applicant_state(appl_id, app_id, tables) for app_id, appl_id in pairs]

    processed = [0]
    start_time = time.time()

    def progress(current, total, result):
        processed[0] = current
        if current % 10 == 0 or current == total:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (total - current) / rate if rate > 0 else 0
            print(f"  {current}/{total} ({100*current/total:.1f}%)  "
                  f"{rate:.1f}/s  ETA: {eta:.0f}s", end="\r")

    results = run_batch(states, max_workers=args.workers, progress_callback=progress)
    print()  # newline after \r progress

    print_batch_summary(results)

    # Save
    out_path = args.output or "results/batch_results.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "application_id": r.get("application_id"),
            "applicant_id": r.get("applicant_id"),
            "final_decision": r.get("final_decision"),
            "approved_amount_nrs": r.get("approved_amount_nrs"),
            "interest_rate_pct": r.get("interest_rate_pct"),
            "interest_tier": r.get("interest_tier"),
            "credit_score": r.get("credit_score"),
            "score_band": r.get("score_band"),
            "income_estimate_monthly": r.get("income_estimate_monthly"),
            "income_confidence": r.get("income_confidence"),
            "compliance_status": r.get("compliance_status"),
            "compliance_flags": str(r.get("compliance_flags", [])),
            "decision_rationale": r.get("decision_rationale"),
            "processing_time_seconds": r.get("processing_time_seconds"),
            "error_message": r.get("error_message"),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[CLI] Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Credit & Lending Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["single", "batch", "summary"],
                        default="batch", help="Run mode")
    parser.add_argument("--data-dir", default="./Track A/structured",
                        help="Directory containing CSV/Parquet files")
    parser.add_argument("--applicant", default=None,
                        help="[single mode] applicant_id to process, e.g. AP-050234")
    parser.add_argument("--application-id", default=None,
                        help="[single mode] Optional application_id override")
    parser.add_argument("--limit", type=int, default=None,
                        help="[batch mode] Max number of applications to process")
    parser.add_argument("--workers", type=int, default=1,
                        help="[batch mode] Number of parallel workers")
    parser.add_argument("--output", default=None,
                        help="Output file path (.json for single, .csv for batch)")

    args = parser.parse_args()

    if args.mode == "single":
        if not args.applicant:
            parser.error("--applicant is required for single mode")
        mode_single(args)
    elif args.mode == "batch":
        mode_batch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

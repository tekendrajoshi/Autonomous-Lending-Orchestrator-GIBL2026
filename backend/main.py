"""
main.py  — FastAPI entry point for the Autonomous Credit Orchestrator
======================================================================
Provides:
  POST /process          — process a single applicant by ID
  POST /batch            — process multiple applicants from uploaded CSVs
  POST /upload           — upload CSV data files to the data store
  GET  /status           — server health check
  GET  /result/{app_id}  — retrieve a previously processed result
  GET  /results          — list all processed results

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import json
import time
import uuid
import traceback
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from orchestrator.data_loader import load_all_tables, get_applicant_state, get_all_applicant_ids
from orchestrator.pipeline import run_single_application, run_batch

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR      = Path(os.getenv("DATA_DIR", "./data"))
RESULTS_DIR   = Path(os.getenv("RESULTS_DIR", "./results"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Global cache of loaded tables (populated after upload or startup)
_tables: dict = {}

# In-memory results store
_results: dict = {}


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Credit & Lending Orchestrator",
    description=(
        "GIBL AI/ML Hackathon 2026 — Track A\n\n"
        "Multi-agent LangGraph pipeline: Income Agent → Score Agent → "
        "Compliance Agent → Decision Agent.\n\n"
        "Provides NRB-compliant credit decisions with full audit trails."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: try to load existing data ────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global _tables
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"[Startup] Loading data from {DATA_DIR}...")
        _tables = load_all_tables(str(DATA_DIR))
    else:
        print("[Startup] No data found. Use POST /upload to upload your CSV files.")


# ── Request / Response models ─────────────────────────────────────────────────
class SingleApplicationRequest(BaseModel):
    applicant_id: str
    application_id: Optional[str] = None

class BatchRequest(BaseModel):
    max_workers: Optional[int] = 1
    limit: Optional[int] = None   # Limit number of applications (None = all)


class ApplicationResult(BaseModel):
    application_id: str
    applicant_id: str
    final_decision: Optional[str]
    approved_amount_nrs: Optional[float]
    interest_rate_pct: Optional[float]
    interest_tier: Optional[str]
    credit_score: Optional[int]
    score_band: Optional[str]
    income_estimate_monthly: Optional[float]
    income_confidence: Optional[float]
    compliance_status: Optional[str]
    compliance_flags: Optional[List[str]]
    decision_rationale: Optional[str]
    processing_time_seconds: Optional[float]
    error_message: Optional[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
def health_check():
    """Server health and loaded data summary."""
    summary = {}
    for name, df in _tables.items():
        summary[name] = len(df) if df is not None else 0
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "loaded_tables": summary,
        "results_count": len(_results),
    }


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more CSV/Parquet files.
    Expected filenames match the Track A dataset:
      - applicant_profiles.csv
      - loan_applications.csv
      - mobile_money_transactions.parquet (or .csv)
      - remittance_records.parquet (or .csv)
      - utility_payments.parquet (or .csv)
      - cooperative_members.csv
      - cooperative_sales.csv
      - document_registry.csv
    """
    global _tables
    saved = []

    for file in files:
        dest = DATA_DIR / file.filename
        contents = await file.read()
        dest.write_bytes(contents)
        saved.append(file.filename)
        print(f"[Upload] Saved {file.filename} ({len(contents):,} bytes)")

    # Reload all tables
    _tables = load_all_tables(str(DATA_DIR))

    return {
        "saved_files": saved,
        "loaded_tables": {k: len(v) if v is not None else 0 for k, v in _tables.items()},
        "message": "Files uploaded and data loaded successfully.",
    }


@app.post("/process", response_model=ApplicationResult)
def process_single(req: SingleApplicationRequest):
    """
    Run the full 4-agent pipeline for a single applicant.
    """
    if not _tables:
        raise HTTPException(
            status_code=400,
            detail="No data loaded. Use POST /upload to upload your CSV files first.",
        )

    # Resolve application_id
    application_id = req.application_id
    if not application_id:
        loans = _tables.get("loans")
        if loans is not None:
            mask = loans["applicant_id"].astype(str) == str(req.applicant_id)
            matches = loans[mask]
            if not matches.empty:
                application_id = str(matches.iloc[0]["application_id"])
        if not application_id:
            application_id = f"LA-MANUAL-{req.applicant_id}"

    initial_state = get_applicant_state(
        applicant_id=req.applicant_id,
        application_id=application_id,
        tables=_tables,
    )

    if initial_state["profiles_data"] is None and initial_state["loans_data"] is None:
        raise HTTPException(
            status_code=404,
            detail=f"Applicant {req.applicant_id} not found in the loaded data.",
        )

    result = run_single_application(initial_state)
    _results[application_id] = result

    return _state_to_result(result)


@app.post("/batch")
def process_batch(req: BatchRequest, background_tasks: BackgroundTasks):
    """
    Process all applications in the loaded dataset (or a limited subset).
    Runs asynchronously in the background.
    Returns a job_id to track progress.
    """
    if not _tables:
        raise HTTPException(
            status_code=400,
            detail="No data loaded. Use POST /upload to upload your CSV files first.",
        )

    pairs = get_all_applicant_ids(_tables)
    if not pairs:
        raise HTTPException(status_code=400, detail="No loan applications found.")

    if req.limit:
        pairs = pairs[: req.limit]

    job_id = str(uuid.uuid4())[:8]
    print(f"[Batch] Starting batch job {job_id} for {len(pairs)} applications...")

    background_tasks.add_task(_run_batch_job, job_id, pairs, req.max_workers or 1)

    return {
        "job_id": job_id,
        "total_applications": len(pairs),
        "message": f"Batch processing started for {len(pairs)} applications. "
                   f"Poll GET /results to see completed results.",
    }


def _run_batch_job(job_id: str, pairs: list, max_workers: int):
    """Background task that processes all applications and saves results."""
    states = [
        get_applicant_state(appl_id, app_id, _tables)
        for app_id, appl_id in pairs
    ]

    def _progress(current, total, result):
        if current % 50 == 0 or current == total:
            print(f"[Batch {job_id}] {current}/{total} processed")

    results = run_batch(states, max_workers=max_workers, progress_callback=_progress)

    for result in results:
        app_id = result.get("application_id", "unknown")
        _results[app_id] = result

    # Save to results CSV
    out_path = RESULTS_DIR / f"batch_{job_id}.csv"
    _save_results_csv(results, out_path)
    print(f"[Batch {job_id}] Complete. Results saved to {out_path}")


@app.get("/results")
def list_results(limit: int = 50, decision_filter: Optional[str] = None):
    """List all processed results, optionally filtered by decision."""
    all_res = [_state_to_result(r) for r in _results.values()]
    if decision_filter:
        all_res = [r for r in all_res if r.final_decision == decision_filter]
    # Summary counts
    from collections import Counter
    counts = Counter(r.final_decision for r in all_res)
    return {
        "total": len(all_res),
        "decision_summary": dict(counts),
        "results": [r.dict() for r in all_res[:limit]],
    }


@app.get("/result/{application_id}")
def get_result(application_id: str):
    """Retrieve the full result for a specific application, including audit trail."""
    if application_id not in _results:
        raise HTTPException(status_code=404, detail=f"No result for {application_id}")
    return _results[application_id]


@app.get("/audit/{application_id}")
def get_audit_trail(application_id: str):
    """Retrieve the full audit trail for a specific application."""
    if application_id not in _results:
        raise HTTPException(status_code=404, detail=f"No result for {application_id}")
    result = _results[application_id]
    return {
        "application_id": application_id,
        "audit_trail": result.get("audit_trail", []),
        "shap_explanation": result.get("shap_explanation", {}),
        "compliance_audit_trail": result.get("compliance_audit_trail", []),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_to_result(state: dict) -> ApplicationResult:
    return ApplicationResult(
        application_id=str(state.get("application_id", "")),
        applicant_id=str(state.get("applicant_id", "")),
        final_decision=state.get("final_decision"),
        approved_amount_nrs=state.get("approved_amount_nrs"),
        interest_rate_pct=state.get("interest_rate_pct"),
        interest_tier=state.get("interest_tier"),
        credit_score=state.get("credit_score"),
        score_band=state.get("score_band"),
        income_estimate_monthly=state.get("income_estimate_monthly"),
        income_confidence=state.get("income_confidence"),
        compliance_status=state.get("compliance_status"),
        compliance_flags=state.get("compliance_flags"),
        decision_rationale=state.get("decision_rationale"),
        processing_time_seconds=state.get("processing_time_seconds"),
        error_message=state.get("error_message"),
    )


def _save_results_csv(results: list[dict], path: Path):
    """Save results to a CSV for downstream analysis."""
    rows = [_state_to_result(r).dict() for r in results]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[Results] Saved {len(rows)} rows to {path}")


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

"""
orchestrator/data_loader.py
============================
Utility module for loading, cleaning, and slicing the Track A CSV/Parquet
files that the user uploads.  Also exposes a function to build a per-applicant
AgentState from the full dataframes (one row lookup by applicant_id).
"""

from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# ── Noise-resistant helpers ───────────────────────────────────────────────────

def _clean_amount(series: pd.Series) -> pd.Series:
    """Coerce amount columns that may have 'Rs. 200,000' noise strings."""
    return (
        series.astype(str)
        .str.replace("Rs.", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
    )


def load_all_tables(data_dir: str) -> dict[str, Optional[pd.DataFrame]]:
    """
    Load all Track A tables from a directory.
    Accepts CSV files only (the user uploads CSVs through the UI).
    Parquet files are also supported if present.
    Returns a dict keyed by table name.
    """
    data_dir = Path(data_dir)
    tables = {}

    def _try_load(name: str, filename: str, parquet: bool = False) -> Optional[pd.DataFrame]:
        path = data_dir / filename
        if not path.exists():
            print(f"[DataLoader] {filename} not found — skipping.")
            return None
        try:
            if parquet or filename.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, low_memory=False)
            print(f"[DataLoader] Loaded {name}: {len(df):,} rows × {len(df.columns)} cols")
            return df
        except Exception as e:
            print(f"[DataLoader] Failed to load {name}: {e}")
            return None

    # Try CSV first, then parquet for the large tables
    tables["profiles"]     = _try_load("profiles",     "applicant_profiles.csv")
    tables["loans"]        = _try_load("loans",        "loan_applications.csv")
    df_tx = _try_load("transactions", "mobile_money_transactions.parquet", parquet=True)
    tables["transactions"] = df_tx if df_tx is not None else _try_load("transactions", "mobile_money_transactions.csv")

    df_rem = _try_load("remittances",  "remittance_records.parquet", parquet=True)
    tables["remittances"] = df_rem if df_rem is not None else _try_load("remittances", "remittance_records.csv")

    df_util = _try_load("utilities",    "utility_payments.parquet", parquet=True)
    tables["utilities"] = df_util if df_util is not None else _try_load("utilities", "utility_payments.csv")
    tables["coop_members"] = _try_load("coop_members", "cooperative_members.csv")
    tables["coop_sales"]   = _try_load("coop_sales",   "cooperative_sales.csv")
    tables["documents"]    = _try_load("documents",    "document_registry.csv")

    # Clean requested_amount_nrs noise
    if tables["loans"] is not None and "requested_amount_nrs" in tables["loans"].columns:
        tables["loans"]["requested_amount_clean"] = _clean_amount(
            tables["loans"]["requested_amount_nrs"]
        ).abs()

    return tables


def get_applicant_state(
    applicant_id: str,
    application_id: str,
    tables: dict,
) -> dict:
    """
    Build the AgentState seed for one applicant by slicing all loaded tables.
    Returns a dict ready to be passed as the initial state to the LangGraph pipeline.
    """

    def _row_to_dict(df: Optional[pd.DataFrame], key_col: str, key_val: str) -> Optional[dict]:
        if df is None:
            return None
        mask = df[key_col].astype(str) == str(key_val)
        rows = df[mask]
        if rows.empty:
            return None
        # Replace NaN with None for JSON serialisability
        return rows.iloc[0].where(rows.iloc[0].notna(), other=None).to_dict()

    def _rows_to_list(df: Optional[pd.DataFrame], key_col: str, key_val: str) -> list[dict]:
        if df is None:
            return []
        mask = df[key_col].astype(str) == str(key_val)
        rows = df[mask]
        if rows.empty:
            return []
        return rows.where(rows.notna(), other=None).to_dict(orient="records")

    profile_row = _row_to_dict(tables.get("profiles"), "applicant_id", applicant_id)
    loan_row    = _row_to_dict(tables.get("loans"),    "applicant_id", applicant_id)

    # Use application_id to look up if applicant has multiple apps
    if loan_row is None:
        loan_row = _row_to_dict(tables.get("loans"), "application_id", application_id)

    return {
        "application_id":     application_id,
        "applicant_id":       applicant_id,
        "profiles_data":      profile_row,
        "loans_data":         loan_row,
        "transactions_data":  _rows_to_list(tables.get("transactions"), "applicant_id", applicant_id),
        "remittances_data":   _rows_to_list(tables.get("remittances"),  "applicant_id", applicant_id),
        "utilities_data":     _rows_to_list(tables.get("utilities"),    "applicant_id", applicant_id),
        "coop_members_data":  _row_to_dict(tables.get("coop_members"),  "applicant_id", applicant_id),
        "coop_sales_data":    _rows_to_list(tables.get("coop_sales"),   "applicant_id", applicant_id),
        # Outputs initialised to None
        "income_estimate_monthly": None,
        "income_confidence": None,
        "credit_score": None,
        "score_band": None,
        "shap_explanation": None,
        "pre_compliance_passed": None,
        "pre_compliance_flags": [],
        "pre_compliance_details": None,
        "compliance_status": None,
        "compliance_flags": [],
        "compliance_modifications": [],
        "compliance_audit_trail": None,
        "recommended_loan_amount": None,
        "final_decision": None,
        "approved_amount_nrs": None,
        "interest_rate_pct": None,
        "interest_tier": None,
        "decision_rationale": None,
        "audit_trail": None,
        "error_message": None,
        "processing_time_seconds": None,
    }


def get_all_applicant_ids(tables: dict) -> list[tuple[str, str]]:
    """
    Returns a list of (application_id, applicant_id) tuples for all applications
    in the loaded loan_applications table.
    """
    loans = tables.get("loans")
    if loans is None:
        return []
    pairs = []
    for _, row in loans.iterrows():
        app_id = str(row.get("application_id", ""))
        appl_id = str(row.get("applicant_id", ""))
        if app_id and appl_id:
            pairs.append((app_id, appl_id))
    return pairs

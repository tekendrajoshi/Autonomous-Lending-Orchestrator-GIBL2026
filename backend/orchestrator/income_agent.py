"""
orchestrator/income_agent.py
=============================
Income Agent — Stream 1 of the pipeline.

Aggregates income signals from eSewa/Khalti wallets, cooperative records,
remittance history, and utility payments to build an income profile.
No payslips needed.

Features engineered (must match the training notebook exactly):
    remittance_monthly_avg, esewa_net_monthly, cooperative_monthly_sales,
    utility_avg_bill_nrs, land_area_ropani, income_signal_count,
    remittance_regularity_score, esewa_tx_count_6months,
    overall_on_time_rate, util_arrears_total_nrs, coop_tenure_years,
    wallet_monthly_inflow, coop_debt_burden, elec_on_time_rate,
    has_utility_arrears, is_ghost_member,
    occupation_en (categorical), rural_urban (categorical)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from orchestrator.state import AgentState

# ── Optional: load the trained income model if the .pkl exists ──────────────
_INCOME_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "trained_income_model.pkl"
)
_income_model = None
_confidence_model = None

def _load_income_models():
    global _income_model, _confidence_model
    if _income_model is not None:
        return
    try:
        import joblib
        if os.path.exists(_INCOME_MODEL_PATH):
            bundle = joblib.load(_INCOME_MODEL_PATH)
            if isinstance(bundle, dict):
                _income_model = bundle.get("income_model")
                _confidence_model = bundle.get("confidence_model")
            else:
                _income_model = bundle
            print(f"[IncomeAgent] Loaded trained model from {_INCOME_MODEL_PATH}")
        else:
            print(f"[IncomeAgent] No model file at {_INCOME_MODEL_PATH} — using rule-based fallback.")
    except Exception as e:
        print(f"[IncomeAgent] Model load failed: {e} — using rule-based fallback.")


# ── Feature Engineering ──────────────────────────────────────────────────────

def _build_income_features(state: AgentState) -> dict:
    """
    Replicates the mathematical aggregation pipeline from income_agent.ipynb.
    Returns a single feature dict matching the model's expected columns.
    """
    profile = state.get("profiles_data") or {}
    transactions = state.get("transactions_data") or []
    remittances = state.get("remittances_data") or []
    utilities = state.get("utilities_data") or []
    coop_member = state.get("coop_members_data") or {}
    coop_sales = state.get("coop_sales_data") or []

    # ── STREAM 1: Mobile Money (Wallet Credits) ──────────────────────────────
    esewa_net_monthly = 0.0
    wallet_monthly_inflow = 0.0
    esewa_tx_count_6months = 0

    if transactions:
        tx_df = pd.DataFrame(transactions)
        # Clean amount column (may have comma-formatted strings)
        tx_df["amount_clean"] = (
            tx_df["amount_nrs"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        tx_df["amount_clean"] = pd.to_numeric(tx_df["amount_clean"], errors="coerce").fillna(0)

        credits = tx_df[tx_df["direction"] == "credit"]["amount_clean"]
        debits = tx_df[tx_df["direction"] == "debit"]["amount_clean"]

        total_credit = credits.sum()
        total_debit = debits.sum()

        # Net monthly over 6-month window
        esewa_net_monthly = max((total_credit - total_debit) / 6, 0.0)
        wallet_monthly_inflow = total_credit / 6
        esewa_tx_count_6months = len(tx_df)

    # ── STREAM 2: Remittances ────────────────────────────────────────────────
    remittance_monthly_avg = 0.0
    remittance_regularity_score = 0.0

    if remittances:
        rem_df = pd.DataFrame(remittances)
        # Filter out impossible exchange rates (10x outlier)
        if "exchange_rate" in rem_df.columns and "foreign_currency_code" in rem_df.columns:
            FX_REF = {"USD": 133.5, "QAR": 36.6, "SAR": 35.5, "AED": 36.3,
                      "MYR": 28.5, "KRW": 0.10, "JPY": 0.85, "INR": 1.60}
            ref_rates = rem_df["foreign_currency_code"].map(FX_REF).fillna(133.5)
            rem_df = rem_df[rem_df["exchange_rate"] <= (10 * ref_rates)]

        # Remove duplicates
        if "_noise_duplicate" in rem_df.columns:
            rem_df = rem_df[~rem_df["_noise_duplicate"].fillna(False)]

        total_remit = rem_df["amount_nrs"].sum() if "amount_nrs" in rem_df.columns else 0.0
        remittance_monthly_avg = total_remit / 6

        # Regularity: how many of the 6 months had a transfer?
        if "received_date_ad" in rem_df.columns:
            months_with_remit = rem_df["received_date_ad"].str[:7].nunique()
            remittance_regularity_score = min(months_with_remit / 6, 1.0)

    # ── STREAM 3: Cooperative Sales ──────────────────────────────────────────
    cooperative_monthly_sales = 0.0
    coop_tenure_years = 0.0
    coop_debt_burden = 0.0
    is_ghost_member = 0

    if coop_member:
        membership_year = coop_member.get("membership_year_bs", 2080)
        coop_tenure_years = max(2081 - int(membership_year), 0)
        outstanding_loan = float(coop_member.get("outstanding_loan_nrs", 0) or 0)
        share_value = float(coop_member.get("total_share_value_nrs", 1000) or 1000)
        coop_debt_burden = outstanding_loan / (share_value + 1e-5)
    else:
        # Ghost member: profile says cooperative_member=True but no record
        if profile.get("cooperative_member") is True:
            is_ghost_member = 1

    if coop_sales:
        sales_df = pd.DataFrame(coop_sales)
        total_sales = sales_df["total_amount_nrs"].sum() if "total_amount_nrs" in sales_df.columns else 0.0
        n_years = max(coop_tenure_years, 1)
        cooperative_monthly_sales = total_sales / (n_years * 12)

    # ── STREAM 4: Utility Payments ───────────────────────────────────────────
    utility_avg_bill_nrs = 0.0
    overall_on_time_rate = 0.0
    util_arrears_total_nrs = 0.0
    has_utility_arrears = 0
    elec_on_time_rate = 0.0

    if utilities:
        util_df = pd.DataFrame(utilities)
        # Drop noise rows
        if "_noise_negative_bill" in util_df.columns:
            util_df = util_df[~util_df["_noise_negative_bill"].fillna(False)]
        if "_noise_forced_unpaid" in util_df.columns:
            util_df = util_df[~util_df["_noise_forced_unpaid"].fillna(False)]

        if "bill_amount_nrs" in util_df.columns:
            util_df["bill_amount_nrs"] = pd.to_numeric(
                util_df["bill_amount_nrs"], errors="coerce"
            ).abs()
            utility_avg_bill_nrs = util_df["bill_amount_nrs"].mean()

        if "cumulative_on_time_rate" in util_df.columns:
            overall_on_time_rate = util_df["cumulative_on_time_rate"].mean()
            elec_df = util_df[util_df.get("utility_type", pd.Series()).eq("electricity")]
            if len(elec_df) > 0:
                elec_on_time_rate = elec_df["cumulative_on_time_rate"].mean()
            else:
                elec_on_time_rate = overall_on_time_rate

        if "outstanding_arrears_nrs" in util_df.columns:
            util_arrears_total_nrs = util_df["outstanding_arrears_nrs"].sum()
            has_utility_arrears = 1 if util_arrears_total_nrs > 0 else 0

    # ── Meta-feature: How many income signals does this applicant have? ──────
    income_signal_count = sum([
        1 if esewa_net_monthly > 0 else 0,
        1 if remittance_monthly_avg > 0 else 0,
        1 if cooperative_monthly_sales > 0 else 0,
    ])

    land_area_ropani = float(profile.get("land_area_ropani", 0) or 0)
    occupation_en = str(profile.get("occupation_en", "Daily Wage Worker") or "Daily Wage Worker")
    rural_urban = str(profile.get("rural_urban", "rural") or "rural")

    return {
        "remittance_monthly_avg": remittance_monthly_avg,
        "esewa_net_monthly": esewa_net_monthly,
        "cooperative_monthly_sales": cooperative_monthly_sales,
        "utility_avg_bill_nrs": utility_avg_bill_nrs,
        "land_area_ropani": land_area_ropani,
        "income_signal_count": income_signal_count,
        "remittance_regularity_score": remittance_regularity_score,
        "esewa_tx_count_6months": esewa_tx_count_6months,
        "overall_on_time_rate": overall_on_time_rate,
        "util_arrears_total_nrs": util_arrears_total_nrs,
        "coop_tenure_years": coop_tenure_years,
        "wallet_monthly_inflow": wallet_monthly_inflow,
        "coop_debt_burden": coop_debt_burden,
        "elec_on_time_rate": elec_on_time_rate,
        "has_utility_arrears": has_utility_arrears,
        "is_ghost_member": is_ghost_member,
        "occupation_en": occupation_en,
        "rural_urban": rural_urban,
    }


def _rule_based_income(features: dict) -> tuple[float, float]:
    """
    Mathematical fallback when no trained model is available.
    Mirrors the reverse-engineered logic from income_agent.ipynb (Cell 6).
    Returns (income_estimate_monthly, income_confidence).
    """
    income = 0.0
    sources = 0

    # Weighted blend of income streams
    remit = features["remittance_monthly_avg"]
    wallet = features["wallet_monthly_inflow"]
    coop = features["cooperative_monthly_sales"]
    land = features["land_area_ropani"]

    if remit > 0:
        income += remit * 0.9       # High reliability; direct cash transfer
        sources += 1
    if wallet > 0:
        income += wallet * 0.6      # Net inflow proxy; some noise
        sources += 1
    if coop > 0:
        income += coop * 0.8        # Seasonal; moderate weight
        sources += 1

    # Land proxy (subsistence farming ≈ 3,000/ropani/month)
    if land > 0 and income == 0:
        income += land * 3000
        sources += 1

    # Floor at minimum wage (NRs 15,000/month for Nepal 2024)
    income = max(income, 8000.0)
    income = min(income, 200_000.0)

    # Confidence: based on number of corroborating signals and on-time rate
    on_time = features["overall_on_time_rate"]
    base_confidence = 0.30 + (sources * 0.15) + (on_time * 0.15)
    base_confidence = round(min(base_confidence, 0.97), 2)
    base_confidence = max(base_confidence, 0.05)

    return round(income, 0), base_confidence


# ── LangGraph Node ───────────────────────────────────────────────────────────

def income_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for the Income Agent.
    Reads applicant data from state, outputs income_estimate_monthly
    and income_confidence back to the shared clipboard.
    """
    print(f"\n[IncomeAgent] Processing applicant: {state.get('applicant_id')}")
    _load_income_models()

    try:
        features = _build_income_features(state)

        if _income_model is not None and _confidence_model is not None:
            # ── ML path ──────────────────────────────────────────────────────
            NUMERIC_FEATURES = [
                "remittance_monthly_avg", "esewa_net_monthly", "cooperative_monthly_sales",
                "utility_avg_bill_nrs", "land_area_ropani", "income_signal_count",
                "remittance_regularity_score", "esewa_tx_count_6months",
                "overall_on_time_rate", "util_arrears_total_nrs", "coop_tenure_years",
                "wallet_monthly_inflow", "coop_debt_burden", "elec_on_time_rate",
                "has_utility_arrears", "is_ghost_member",
            ]
            CAT_FEATURES = ["occupation_en", "rural_urban"]

            row = pd.DataFrame([features])
            for col in NUMERIC_FEATURES:
                row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)
            for col in CAT_FEATURES:
                row[col] = row[col].astype("category")

            income_est = float(
                np.clip(_income_model.predict(row[NUMERIC_FEATURES + CAT_FEATURES]), 3000, 200000)[0]
            )
            confidence = float(
                np.clip(_confidence_model.predict(row[NUMERIC_FEATURES + CAT_FEATURES]), 0.05, 0.97)[0]
            )
            income_est = round(income_est, 0)
            confidence = round(confidence, 2)
            print(f"[IncomeAgent] ML prediction → income={income_est:,.0f} NRs/mo  confidence={confidence:.2f}")
        else:
            # ── Rule-based fallback ───────────────────────────────────────────
            income_est, confidence = _rule_based_income(features)
            print(f"[IncomeAgent] Rule-based → income={income_est:,.0f} NRs/mo  confidence={confidence:.2f}")

        return {
            "income_estimate_monthly": income_est,
            "income_confidence": confidence,
        }

    except Exception as exc:
        print(f"[IncomeAgent] ERROR: {exc}")
        # Conservative fallback
        return {
            "income_estimate_monthly": 15000.0,
            "income_confidence": 0.10,
            "error_message": f"IncomeAgent: {exc}",
        }

"""
orchestrator/compliance_agent.py
==================================
Compliance Agent — Stream 3 of the pipeline.

Validates the proposed loan against NRB Unified Directives 2080:
  • Pre-checks  (KYC/AML/Blacklist/Duplicate) — blocks before scoring
  • Post-checks (LTI/DSCR/LTV/Sector/Interest) — run after score

Wraps the existing compliance_engine.py from compliance_agent/ by
translating AgentState into a LoanApplication Pydantic model.
"""

from __future__ import annotations

import os
import sys
import importlib
from typing import Optional

# Add the compliance_agent directory to the Python path so we can import
# its modules (models, rules, services) cleanly.
_COMPLIANCE_DIR = os.path.join(os.path.dirname(__file__), "..", "compliance_agent")
if _COMPLIANCE_DIR not in sys.path:
    sys.path.insert(0, _COMPLIANCE_DIR)

from orchestrator.state import AgentState


def _build_loan_application(state: AgentState):
    """
    Converts the AgentState clipboard into a LoanApplication Pydantic model
    expected by the compliance engine.
    """
    from models.request_model import LoanApplication  # noqa: E402

    profile = state.get("profiles_data") or {}
    loans   = state.get("loans_data") or {}

    # ── Requested amount — handle string noise ────────────────────────────────
    raw_amount = loans.get("requested_amount_nrs", 200000)
    try:
        requested_amount = float(
            str(raw_amount).replace("Rs.", "").replace(",", "").strip()
        )
    except (ValueError, TypeError):
        requested_amount = 200000.0
    if requested_amount <= 0:
        requested_amount = abs(requested_amount) or 200000.0

    # ── Loan tenure — coerce to valid literal ─────────────────────────────────
    raw_tenure = loans.get("requested_tenure_months", 24)
    try:
        tenure = int(raw_tenure)
    except (ValueError, TypeError):
        tenure = 24
    VALID_TENURES = {6, 12, 18, 24, 36}
    if tenure not in VALID_TENURES:
        # Snap to nearest
        tenure = min(VALID_TENURES, key=lambda t: abs(t - tenure))

    # ── Monthly income from income agent ─────────────────────────────────────
    monthly_income = float(state.get("income_estimate_monthly") or 15000)
    monthly_income = max(3000.0, min(200000.0, monthly_income))

    # ── Income confidence ─────────────────────────────────────────────────────
    income_confidence = float(state.get("income_confidence") or 0.30)
    income_confidence = max(0.05, min(0.97, income_confidence))

    # ── Credit score from score agent ─────────────────────────────────────────
    credit_score = int(state.get("credit_score") or 400)
    credit_score = max(300, min(850, credit_score))

    # ── Proposed EMI (flat monthly installment) ───────────────────────────────
    proposed_emi = requested_amount / tenure

    # ── Interest rate — handle out-of-corridor noise ─────────────────────────
    raw_rate = loans.get("interest_rate_pct", 12.5)
    try:
        interest_rate = float(str(raw_rate).strip())
    except (ValueError, TypeError):
        interest_rate = 12.5
    if interest_rate <= 0 or interest_rate > 100:
        interest_rate = 12.5

    # ── Collateral ────────────────────────────────────────────────────────────
    collateral_type = str(loans.get("collateral_type", "none") or "none")
    VALID_COLL = {"none", "land", "residential_land", "commercial_property", "fixed_deposit"}
    if collateral_type not in VALID_COLL:
        collateral_type = "land" if collateral_type else "none"

    collateral_value = float(loans.get("collateral_value_nrs", 0) or 0)

    # ── KYC documents ─────────────────────────────────────────────────────────
    has_citizenship = bool(loans.get("has_citizenship", True))
    has_pan = bool(loans.get("has_pan", False))
    has_tax = bool(loans.get("has_tax_clearance", False))
    has_lalpurja = collateral_type in ("land", "residential_land")

    # ── Loan purpose ──────────────────────────────────────────────────────────
    loan_purpose = str(loans.get("loan_purpose", "agricultural_input") or "agricultural_input")
    VALID_PURPOSES = {
        "agricultural_input", "small_trade", "livestock_purchase", "home_repair",
        "microenterprise_startup", "education", "agri_land_development",
        "irrigation_equipment", "vehicle_purchase", "emergency_medical",
    }
    if loan_purpose not in VALID_PURPOSES:
        loan_purpose = "agricultural_input"

    # doc_completeness_score — read from loan table and clamp to valid range
    # BUGFIX: was always None because default was None and no coercion was done.
    doc_score_raw = loans.get("doc_completeness_score", None)
    if doc_score_raw is not None:
        try:
            doc_score = float(doc_score_raw)
            if not (0.0 <= doc_score <= 1.0):
                doc_score = None  # invalid range
        except (ValueError, TypeError):
            doc_score = None
    else:
        doc_score = None

    existing_loans = int(loans.get("existing_loan_count", 0) or 0)
    existing_loans = max(0, min(3, existing_loans))

    credit_bureau = loans.get("credit_bureau_score", None)
    if credit_bureau is not None:
        try:
            credit_bureau = float(credit_bureau)
            if credit_bureau < 300 or credit_bureau > 850:
                credit_bureau = None
        except (ValueError, TypeError):
            credit_bureau = None

    app_date = str(loans.get("application_date_ad", "2024-04-15") or "2024-04-15")

    return LoanApplication(
        applicant_name=str(profile.get("full_name_en", "Unknown")),
        requested_amount_nrs=requested_amount,
        loan_term_months=tenure,
        loan_purpose=loan_purpose,
        interest_rate_pct=interest_rate,
        existing_loan_count=existing_loans,
        monthly_income_nrs=monthly_income,
        income_confidence=income_confidence,
        proposed_emi_nrs=proposed_emi,
        credit_score=credit_score,
        credit_bureau_score=credit_bureau,
        doc_completeness_score=doc_score,
        has_citizenship=has_citizenship,
        has_pan=has_pan,
        has_tax_clearance=has_tax,
        has_lalpurja=has_lalpurja,
        collateral_type=collateral_type,
        collateral_value_nrs=collateral_value,
        application_date=app_date,
    )


# ── LangGraph Node ────────────────────────────────────────────────────────────

def compliance_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for the NRB Compliance Agent.
    Wraps the full compliance engine and writes results to state.
    """
    print(f"\n[ComplianceAgent] Processing: {state.get('application_id')}")

    try:
        from services.compliance_engine import run_compliance  # noqa: E402

        loan_app = _build_loan_application(state)
        response = run_compliance(loan_app)

        return {
            "compliance_status":      response.compliance_status,
            "compliance_flags":       response.compliance_flags or [],
            "compliance_modifications": response.modifications or [],
            "compliance_audit_trail": response.audit_trail or [],
            "recommended_loan_amount": response.recommended_loan_amount,
        }

    except Exception as exc:
        print(f"[ComplianceAgent] ERROR: {exc}")
        # Fallback: basic rule checks inline
        return _fallback_compliance(state, str(exc))


def _fallback_compliance(state: AgentState, err_msg: str = "") -> dict:
    """
    Minimal NRB checks when the full compliance engine fails to import.
    Covers: Blacklist, AML, LTI, DSCR basics.
    """
    loans = state.get("loans_data") or {}
    flags = []
    mods = []

    monthly_income = float(state.get("income_estimate_monthly") or 15000)

    raw_amount = loans.get("requested_amount_nrs", 200000)
    try:
        requested = float(str(raw_amount).replace("Rs.", "").replace(",", "").strip())
    except (ValueError, TypeError):
        requested = 200000.0

    # Blacklist → immediate reject
    if loans.get("nrb_blacklist_flag") is True:
        return {
            "compliance_status": "reject",
            "compliance_flags": ["NRB_BLACKLIST"],
            "compliance_modifications": [],
            "compliance_audit_trail": [{"rule": "NRB-BLACKLIST", "result": "REJECT"}],
            "recommended_loan_amount": None,
            "error_message": f"Fallback mode: {err_msg}",
        }

    # AML
    if loans.get("aml_flag") is True:
        flags.append("AML_REVIEW")

    # LTI check (agricultural cap = 36x monthly income)
    loan_purpose = str(loans.get("loan_purpose", "") or "")
    AGRI = {"agricultural_input", "livestock_purchase", "agri_land_development", "irrigation_equipment"}
    if loan_purpose in AGRI and monthly_income > 0:
        lti = requested / monthly_income
        if lti > 36:
            flags.append("LTI_EXCEEDED")
            cap = monthly_income * 36
            mods.append(f"LTI_EXCEEDED: capped at NRs {cap:,.0f}")
            requested = min(requested, cap)

    # DSCR: EMI > 60% of income → hard reject; 50-60% → conditional warning
    raw_tenure = loans.get("requested_tenure_months", 24) or 24
    try:
        tenure = int(raw_tenure)
    except (ValueError, TypeError):
        tenure = 24
    emi = requested / max(tenure, 1)
    if emi > 0.60 * monthly_income:
        flags.append("DSCR_EXCEEDED")      # hard reject path in decision agent
    elif emi > 0.50 * monthly_income:
        flags.append("DSCR_WARNING")        # conditional path

    # Uncollateralized high-value
    if requested > 500000 and str(loans.get("collateral_type", "none")) == "none":
        flags.append("UNCOLLATERALIZED_HIGH_VALUE")

    # Interest rate corridor
    try:
        rate = float(loans.get("interest_rate_pct", 12.5) or 12.5)
        if rate > 0 and not (10.0 <= rate <= 15.0):
            flags.append("INTEREST_RATE_OOR")
    except (ValueError, TypeError):
        pass

    status = "flag" if flags else "pass"

    return {
        "compliance_status": status,
        "compliance_flags": flags,
        "compliance_modifications": mods,
        "compliance_audit_trail": [{"rule": "FALLBACK", "flags": flags}],
        "recommended_loan_amount": requested if not flags else min(requested, monthly_income * 36),
        "error_message": f"Fallback mode: {err_msg}" if err_msg else None,
    }

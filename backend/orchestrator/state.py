"""
orchestrator/state.py
=====================
Shared state definition for the LangGraph pipeline.
Think of this as the "clipboard" that every desk worker (agent) reads from
and writes to as the application moves through the pipeline.
"""

from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict):
    # ─── Application Identifiers ──────────────────────────────────────────────
    application_id: str              # e.g. "LA-2081-000001"
    applicant_id: str                # e.g. "AP-050234"

    # ─── Raw Input DataFrames (passed as serialisable dicts) ─────────────────
    # All CSV files uploaded by the user are loaded here once and shared.
    profiles_data: Optional[Dict]    # applicant_profiles row as dict
    loans_data: Optional[Dict]       # loan_applications row as dict
    transactions_data: Optional[List[Dict]]   # mobile_money_transactions rows
    remittances_data: Optional[List[Dict]]    # remittance_records rows
    utilities_data: Optional[List[Dict]]      # utility_payments rows
    coop_members_data: Optional[Dict]         # cooperative_members row
    coop_sales_data: Optional[List[Dict]]     # cooperative_sales rows

    # ─── Income Agent Outputs ────────────────────────────────────────────────
    income_estimate_monthly: Optional[float]  # NRs/month, e.g. 45000.0
    income_confidence: Optional[float]        # 0.05 – 0.97

    # ─── Score Agent Outputs ─────────────────────────────────────────────────
    credit_score: Optional[int]               # 300 – 850
    score_band: Optional[str]                 # poor / fair / good / very_good / excellent
    shap_explanation: Optional[Dict]          # {feature: shap_value}

    # ─── Pre-Compliance Agent Outputs ────────────────────────────────────────
    pre_compliance_passed: Optional[bool]
    pre_compliance_flags: List[str]
    pre_compliance_details: Optional[Dict]    # structured audit trail from pre_compliance.py

    # ─── Post-Compliance Agent Outputs ───────────────────────────────────────
    compliance_status: Optional[str]          # "pass" / "flag" / "reject"
    compliance_flags: List[str]               # e.g. ["LTI_EXCEEDED", "AML_REVIEW"]
    compliance_modifications: List[str]       # suggested loan adjustments
    compliance_audit_trail: Optional[List[Dict]]
    recommended_loan_amount: Optional[float]  # after haircut/LTI cap

    # ─── Decision Agent Outputs ──────────────────────────────────────────────
    final_decision: Optional[str]             # "approve" / "conditional" / "refer" / "reject"
    approved_amount_nrs: Optional[float]
    interest_rate_pct: Optional[float]
    interest_tier: Optional[str]              # "base" / "premium" / "subprime"
    decision_rationale: Optional[str]         # human-readable explanation
    audit_trail: Optional[List[Dict]]         # merged full audit trail

    # ─── Pipeline Metadata ───────────────────────────────────────────────────
    error_message: Optional[str]              # last error, if any
    processing_time_seconds: Optional[float]

"""
orchestrator/decision_agent.py
================================
Decision Agent — Terminal node of the LangGraph pipeline.

Synthesises all upstream agent outputs into a final credit decision:
  approve / conditional / refer / reject

Handles disagreements between agents via a weighted resolution protocol:
  • Hard vetoes (blacklist, AML) → always reject
  • Compliance flags override positive income/score signals
  • Score band feeds interest tier selection
  • Low confidence → refer even if score is good

Also produces:
  - approved_amount_nrs (with haircuts applied)
  - interest_rate_pct (NRB corridor: 10-15%)
  - interest_tier (base / premium / subprime)
  - decision_rationale (human-readable explanation for audit)
  - Full merged audit trail
"""

from __future__ import annotations

import json
from typing import Optional

from orchestrator.state import AgentState


# ── Decision thresholds ───────────────────────────────────────────────────────
SCORE_APPROVE_MIN   = 580     # "good" band threshold
SCORE_REFER_MIN     = 450     # "fair" band minimum for referral
CONFIDENCE_REFER    = 0.40    # below this → refer regardless of score
CONFIDENCE_HAIRCUT  = 0.70    # below this → 15% haircut on approved amount

INTEREST_RATE_BASE      = 10.5   # Best rate (excellent credit)
INTEREST_RATE_PREMIUM   = 12.5   # Standard rate
INTEREST_RATE_SUBPRIME  = 14.5   # High-risk rate (NRB ceiling = 15%)
INTEREST_RATE_MAX       = 15.0   # Hard NRB ceiling

# Hard-reject compliance flag codes (immediate REJECT, no loan)
HARD_REJECT_FLAGS = {
    "NRB_BLACKLIST", "DUPLICATE_APPLICATION",
    "SECTOR_EXPOSURE_EXCEEDED", "SECTOR_LIMIT_EXCEEDED",  # both names covered
    "KYC_REJECT", "AML_BLACKLIST",
    "DSCR_EXCEEDED",  # EMI > 60% of income → loan is unaffordable
}

# Conditional flags (loan proceeds with amount / rate modifications)
CONDITIONAL_FLAGS = {
    "LTI_EXCEEDED", "DSCR_WARNING", "LTV_EXCEEDED",
    "INTEREST_RATE_ADJUSTMENT", "UNCOLLATERALIZED_HIGH_VALUE",
    "INTEREST_RATE_OOR", "LOW_INCOME_CONFIDENCE",  # haircut applied, proceed as conditional
}

# Refer flags (send to human officer for review)
# NOTE: LOW_INCOME_CONFIDENCE is NOT here — it is handled by Rule 4 (income_confidence < 0.40)
# and the 0.40–0.70 band is handled via the haircut/CONDITIONAL path.
REFER_FLAGS = {
    "AML_REVIEW", "LOW_CREDIT_SCORE",
    "INCOMPLETE_DOCUMENTS", "DOC_PIPELINE_ERROR",
}


def _select_interest_rate(score_band: str, loan_purpose: str) -> tuple[float, str]:
    """
    Returns (interest_rate_pct, interest_tier) based on credit score band
    and loan purpose, within NRB corridor 10%–15%.
    """
    if score_band in ("excellent", "very_good"):
        return INTEREST_RATE_BASE, "base"
    elif score_band == "good":
        return INTEREST_RATE_PREMIUM, "premium"
    else:
        return INTEREST_RATE_SUBPRIME, "subprime"


def _apply_haircut(amount: float, confidence: float) -> float:
    """Applies a 15% haircut if income confidence is moderate (0.40–0.70)."""
    if confidence < CONFIDENCE_HAIRCUT:
        return round(amount * 0.85, 2)
    return round(amount, 2)


def decision_agent_node(state: AgentState) -> dict:
    """
    LangGraph terminal node for the Decision Agent.
    Reads the full state clipboard and produces the final lending decision.
    """
    print(f"\n[DecisionAgent] Synthesising decision for: {state.get('application_id')}")

    # ── Pull all upstream outputs ─────────────────────────────────────────────
    compliance_status   = state.get("compliance_status") or "flag"
    compliance_flags    = set(state.get("compliance_flags") or [])
    compliance_mods     = state.get("compliance_modifications") or []
    credit_score        = int(state.get("credit_score") or 400)
    score_band          = state.get("score_band") or "poor"
    income_est          = float(state.get("income_estimate_monthly") or 15000)
    income_confidence   = float(state.get("income_confidence") or 0.30)
    recommended_amount  = state.get("recommended_loan_amount")
    loans               = state.get("loans_data") or {}

    # Raw requested amount
    raw_amount = loans.get("requested_amount_nrs", 200000)
    try:
        requested = float(str(raw_amount).replace("Rs.", "").replace(",", "").strip())
    except (ValueError, TypeError):
        requested = 200000.0
    if requested <= 0:
        requested = abs(requested) or 200000.0

    # Use recommended amount from compliance (post haircut/LTI), or requested
    working_amount = float(recommended_amount or requested)

    loan_purpose = str(loans.get("loan_purpose", "agricultural_input") or "agricultural_input")

    rationale_parts = []
    final_decision = None
    approved_amount = None
    interest_rate = None
    interest_tier = None

    # ── RULE 1: Hard Veto Flags → immediate REJECT ───────────────────────────
    hard_reject_hits = compliance_flags & HARD_REJECT_FLAGS
    if hard_reject_hits:
        final_decision = "REJECT"
        rationale_parts.append(
            f"HARD REJECT: Triggered by compliance flag(s): {', '.join(hard_reject_hits)}. "
            f"Per NRB Unified Directives 2080, these violations result in immediate rejection."
        )
        approved_amount = None
        interest_rate = 0.0
        interest_tier = None

    # ── RULE 2: AML Review → REFER ───────────────────────────────────────────
    elif "AML_REVIEW" in compliance_flags:
        final_decision = "REFER"
        rationale_parts.append(
            "REFER: AML screening raised a high-risk flag. "
            "Application routed to AML Compliance Officer for manual review. "
            "Disbursement blocked until cleared."
        )
        approved_amount = None
        interest_rate = 0.0
        interest_tier = None

    # ── RULE 3: Compliance engine explicitly flagged for REFER ────────────────
    # When the engine returns 'refer' or 'flag' AND refer-type flags are present,
    # we must honour that — do NOT silently downgrade to CONDITIONAL.
    elif compliance_status == "refer" or (compliance_flags & REFER_FLAGS):
        refer_hits = compliance_flags & REFER_FLAGS
        final_decision = "REFER"
        rationale_parts.append(
            f"REFER: Manual review required. Active flags: {', '.join(refer_hits) if refer_hits else 'compliance engine decision'}. "
            f"Application routed to loan officer for document/income/KYC verification."
        )
        approved_amount = None
        interest_rate = 0.0
        interest_tier = None

    # ── RULE 4: Very low income confidence → REFER ───────────────────────────
    elif income_confidence < CONFIDENCE_REFER:
        final_decision = "REFER"
        rationale_parts.append(
            f"REFER: Income confidence ({income_confidence:.2f}) is critically low (< {CONFIDENCE_REFER}). "
            f"Income estimate of NRs {income_est:,.0f}/month cannot be sufficiently verified from "
            f"available alternative data. Manual income verification required."
        )
        approved_amount = None
        interest_rate = 0.0
        interest_tier = None

    # ── RULE 5: Poor credit score → REFER ────────────────────────────────────
    elif credit_score < SCORE_REFER_MIN:
        final_decision = "REFER"
        rationale_parts.append(
            f"REFER: Credit score {credit_score} is in the 'Poor' band (< {SCORE_REFER_MIN}). "
            f"Application referred to senior loan officer for manual assessment. "
            f"Applicant may reapply after 90 days with improved financial behaviour."
        )
        approved_amount = None
        interest_rate = 0.0
        interest_tier = None

    # ── RULE 6: Conditional flags present → CONDITIONAL ──────────────────────
    elif compliance_flags & CONDITIONAL_FLAGS or compliance_mods:
        final_decision = "CONDITIONAL"
        # Apply haircut if moderate confidence
        working_amount = _apply_haircut(working_amount, income_confidence)
        interest_rate, interest_tier = _select_interest_rate(score_band, loan_purpose)
        approved_amount = working_amount

        flag_summary = ", ".join(compliance_flags & CONDITIONAL_FLAGS) if compliance_flags & CONDITIONAL_FLAGS else "modifications required"
        rationale_parts.append(
            f"CONDITIONAL APPROVAL: Loan approved at NRs {working_amount:,.0f} "
            f"(modified from requested NRs {requested:,.0f}) subject to: {flag_summary}."
        )
        if compliance_mods:
            rationale_parts.append("Modifications: " + "; ".join(compliance_mods))
        if income_confidence < CONFIDENCE_HAIRCUT:
            rationale_parts.append(
                f"15% haircut applied due to moderate income confidence ({income_confidence:.2f})."
            )

    # ── RULE 7: Score ≥ 580 and compliance passed → APPROVE ──────────────────
    elif credit_score >= SCORE_APPROVE_MIN and compliance_status == "pass":
        final_decision = "APPROVE"
        working_amount = _apply_haircut(working_amount, income_confidence)
        interest_rate, interest_tier = _select_interest_rate(score_band, loan_purpose)
        approved_amount = working_amount
        rationale_parts.append(
            f"APPROVED: Credit score {credit_score} ({score_band}) satisfies lending threshold. "
            f"All NRB compliance checks passed. "
            f"Approved amount: NRs {working_amount:,.0f} at {interest_rate}% p.a. ({interest_tier} tier)."
        )

    # ── RULE 8: Score fair/good, compliance flagged but no hard blocks → CONDITIONAL
    elif credit_score >= SCORE_REFER_MIN:
        final_decision = "CONDITIONAL"
        working_amount = _apply_haircut(working_amount, income_confidence)
        interest_rate, interest_tier = _select_interest_rate(score_band, loan_purpose)
        approved_amount = working_amount
        rationale_parts.append(
            f"CONDITIONAL APPROVAL: Credit score {credit_score} ({score_band}) meets minimum threshold. "
            f"Compliance flagged for review but not a hard block. "
            f"Loan approved subject to applicant meeting stated conditions."
        )
        if compliance_flags:
            rationale_parts.append(f"Active flags: {', '.join(compliance_flags)}.")

    # ── Default: REJECT ────────────────────────────────────────────────────────
    else:
        final_decision = "REJECT"
        rationale_parts.append(
            f"REJECTED: Credit score {credit_score} ({score_band}) below minimum threshold "
            f"and compliance flags present. Loan cannot be approved under current NRB guidelines."
        )
        approved_amount = None
        interest_rate = 0.0
        interest_tier = None

    # ── Build merged audit trail ───────────────────────────────────────────────
    pre_flags = state.get("pre_compliance_flags") or []
    pre_details = state.get("pre_compliance_details") or {}
    compliance_trail = state.get("compliance_audit_trail") or []
    shap_exp = state.get("shap_explanation") or {}

    full_audit = {
        "application_id": state.get("application_id"),
        "applicant_id": state.get("applicant_id"),
        "pipeline_stages": {
            "income_agent": {
                "income_estimate_monthly": income_est,
                "income_confidence": income_confidence,
            },
            "score_agent": {
                "credit_score": credit_score,
                "score_band": score_band,
                "shap_top_features": shap_exp,
            },
            "pre_compliance": {
                "flags": pre_flags,
                "details": pre_details,
            },
            "compliance_agent": {
                "status": compliance_status,
                "flags": list(compliance_flags),
                "modifications": compliance_mods,
                "trail": compliance_trail,
            },
        },
        "decision": {
            "final_decision": final_decision,
            "approved_amount_nrs": approved_amount,
            "interest_rate_pct": interest_rate,
            "interest_tier": interest_tier,
            "rationale": " | ".join(rationale_parts),
        },
    }

    print(f"[DecisionAgent] Final decision: {final_decision.upper()}")
    if approved_amount:
        print(f"[DecisionAgent] Approved amount: NRs {approved_amount:,.0f}")

    return {
        "final_decision":       final_decision.upper(),
        "approved_amount_nrs":  approved_amount,
        "interest_rate_pct":    interest_rate,
        "interest_tier":        interest_tier,
        "decision_rationale":   " | ".join(rationale_parts),
        "audit_trail":          [full_audit],
    }

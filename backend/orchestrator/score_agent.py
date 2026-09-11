"""
orchestrator/score_agent.py
============================
Credit Score Agent — Stream 2 of the pipeline.

Normalises heterogeneous income signals and runs the XGBoost-based
credit scoring model (credit_score_model.joblib) trained on Nepali
alternative data.  Outputs a score (300-850) with SHAP explanations.

Feature set comes from pipeline_score_agent.ipynb's
process_and_combine_test_dataset() function.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from orchestrator.state import AgentState

# ── Model loading ─────────────────────────────────────────────────────────────
_SCORE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "credit_score_model.joblib"
)
_SCORE_MODEL_PATH_FALLBACK = os.path.join(
    os.path.dirname(__file__), "..", "credit_score_model.pkl"
)
_score_model = None


def _load_score_model():
    global _score_model
    if _score_model is not None:
        return
    try:
        import joblib
        import sklearn.compose._column_transformer
        
        # Patch for loading sklearn 1.6.1 model in 1.9.0
        if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
            class _RemainderColsList:
                pass
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

        if os.path.exists(_SCORE_MODEL_PATH):
            _score_model = joblib.load(_SCORE_MODEL_PATH)
            print(f"[ScoreAgent] Loaded model from {_SCORE_MODEL_PATH}")
        elif os.path.exists(_SCORE_MODEL_PATH_FALLBACK):
            _score_model = joblib.load(_SCORE_MODEL_PATH_FALLBACK)
            print(f"[ScoreAgent] Loaded model from {_SCORE_MODEL_PATH_FALLBACK}")
        else:
            print(f"[ScoreAgent] Model not found at {_SCORE_MODEL_PATH} — using rule-based fallback.")
    except Exception as e:
        print(f"[ScoreAgent] Model load error: {e} — using rule-based fallback.")


# ── Feature columns expected by the joblib model ─────────────────────────────
# Derived from pipeline_score_agent.ipynb's target_columns list
# (only the ML-trainable features, excluding ID/split columns)
_SCORE_FEATURES = [
    # Profile features
    "land_area_ropani", "household_size", "existing_loan_count",
    # Income agent outputs
    "income_agent_monthly_est", "income_confidence",
    # Cooperative
    "share_count", "total_share_value_nrs", "last_annual_dividend_nrs",
    "outstanding_loan_nrs", "membership_time_yrs", "num_sales",
    "total_sales_amount_nrs", "has_coop_sales",
    # Remittance aggregates
    "remit_total_amount_nrs", "remit_mean_amount_nrs", "remit_count",
    "remit_num_unique_countries", "remit_min_name_match_score",
    "remit_pct_digital", "remit_income_volatility", "remit_monthly_avg_inflow",
    "has_remittance_history",
    # Wallet aggregates
    "wallet_tx_count", "wallet_total_credit", "wallet_total_debit",
    "wallet_topup_total", "wallet_utility_spend_total",
    "wallet_agri_spend_total", "wallet_loan_repay_total",
    "wallet_monthly_avg_income", "wallet_monthly_avg_utility",
    "wallet_agri_reinvestment_rate", "wallet_cash_retention_rate",
    # Utility aggregates
    "util_tx_count", "util_avg_bill_nrs", "util_avg_days_late",
    "util_late_payment_rate", "util_early_payment_rate",
    "util_severe_late_count", "util_digital_payment_rate",
    "util_arrears_to_bill_ratio", "has_utility_history",
    # Doc/CIB flags
    "doc_completeness_score", "has_cib_history",
]

_CAT_SCORE_FEATURES = ["occupation_en", "education_level", "rural_urban",
                        "kyc_tier", "loan_purpose", "collateral_type",
                        "coop_loan_repayment_status", "membership_status"]


def _build_score_features(state: AgentState) -> pd.DataFrame:
    """
    Constructs the 1-row feature DataFrame expected by the score model.
    Mirrors the aggregation pipeline from pipeline_score_agent.ipynb.
    """
    profile = state.get("profiles_data") or {}
    loans = state.get("loans_data") or {}
    transactions = state.get("transactions_data") or []
    remittances = state.get("remittances_data") or []
    utilities = state.get("utilities_data") or []
    coop_member = state.get("coop_members_data") or {}
    coop_sales = state.get("coop_sales_data") or []

    feat: dict = {}

    # ── Profile features ─────────────────────────────────────────────────────
    feat["land_area_ropani"] = float(profile.get("land_area_ropani", 0) or 0)
    feat["household_size"] = float(profile.get("household_size", 4) or 4)
    feat["occupation_en"] = str(profile.get("occupation_en", "Daily Wage Worker") or "Daily Wage Worker")
    feat["education_level"] = str(profile.get("education_level", "Unknown") or "Unknown")
    feat["rural_urban"] = str(profile.get("rural_urban", "rural") or "rural")
    feat["kyc_tier"] = str(profile.get("kyc_tier", "basic") or "basic")

    # ── Loan features ─────────────────────────────────────────────────────────
    feat["existing_loan_count"] = float(loans.get("existing_loan_count", 0) or 0)
    feat["doc_completeness_score"] = float(loans.get("doc_completeness_score", 0.5) or 0.5)
    feat["loan_purpose"] = str(loans.get("loan_purpose", "agricultural_input") or "agricultural_input")
    feat["collateral_type"] = str(loans.get("collateral_type", "none") or "none")
    feat["has_cib_history"] = 1 if (loans.get("credit_bureau_score") is not None) else 0

    # ── Income agent outputs (written in prior step) ───────────────────────
    feat["income_agent_monthly_est"] = float(state.get("income_estimate_monthly", 15000) or 15000)
    feat["income_confidence"] = float(state.get("income_confidence", 0.30) or 0.30)

    # ── Wallet aggregates ─────────────────────────────────────────────────────
    wallet_tx_count = 0
    wallet_total_credit = 0.0
    wallet_total_debit = 0.0
    wallet_topup_total = 0.0
    wallet_utility_spend_total = 0.0
    wallet_agri_spend_total = 0.0
    wallet_loan_repay_total = 0.0
    wallet_monthly_avg_income = 0.0
    wallet_monthly_avg_utility = 0.0
    wallet_agri_reinvestment_rate = 0.0
    wallet_cash_retention_rate = 0.0

    if transactions:
        tx_df = pd.DataFrame(transactions)
        tx_df["amount_clean"] = (
            tx_df["amount_nrs"].astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        tx_df["amount_clean"] = pd.to_numeric(tx_df["amount_clean"], errors="coerce").fillna(0)

        wallet_tx_count = len(tx_df)
        credits = tx_df[tx_df["direction"] == "credit"]["amount_clean"].sum()
        debits = tx_df[tx_df["direction"] == "debit"]["amount_clean"].sum()
        wallet_total_credit = credits
        wallet_total_debit = debits

        if "transaction_type" in tx_df.columns:
            wallet_topup_total = tx_df[tx_df["transaction_type"] == "wallet_topup"]["amount_clean"].sum()
            wallet_loan_repay_total = tx_df[tx_df["transaction_type"] == "loan_repayment"]["amount_clean"].sum()

        if "counterparty_category" in tx_df.columns:
            wallet_utility_spend_total = tx_df[tx_df["counterparty_category"] == "utility"]["amount_clean"].sum()
            wallet_agri_spend_total = tx_df[tx_df["counterparty_category"] == "agriculture_input"]["amount_clean"].sum()

        wallet_monthly_avg_income = credits / 6
        wallet_monthly_avg_utility = wallet_utility_spend_total / 6
        wallet_agri_reinvestment_rate = wallet_agri_spend_total / (credits + 1e-5)
        wallet_cash_retention_rate = (credits - debits) / (credits + 1e-5)

    feat.update({
        "wallet_tx_count": wallet_tx_count,
        "wallet_total_credit": wallet_total_credit,
        "wallet_total_debit": wallet_total_debit,
        "wallet_topup_total": wallet_topup_total,
        "wallet_utility_spend_total": wallet_utility_spend_total,
        "wallet_agri_spend_total": wallet_agri_spend_total,
        "wallet_loan_repay_total": wallet_loan_repay_total,
        "wallet_monthly_avg_income": wallet_monthly_avg_income,
        "wallet_monthly_avg_utility": wallet_monthly_avg_utility,
        "wallet_agri_reinvestment_rate": wallet_agri_reinvestment_rate,
        "wallet_cash_retention_rate": wallet_cash_retention_rate,
    })

    # ── Remittance aggregates ─────────────────────────────────────────────────
    remit_total = 0.0
    remit_mean = 0.0
    remit_count = 0
    remit_countries = 0
    remit_min_name_match = -1.0
    remit_pct_digital = 0.0
    remit_volatility = 0.0
    remit_monthly_avg = 0.0
    has_remit = 0

    if remittances:
        rem_df = pd.DataFrame(remittances)
        if "_noise_duplicate" in rem_df.columns:
            rem_df = rem_df[~rem_df["_noise_duplicate"].fillna(False)]

        if "amount_nrs" in rem_df.columns:
            remit_total = rem_df["amount_nrs"].sum()
            remit_mean = rem_df["amount_nrs"].mean()
            remit_count = len(rem_df)
            remit_monthly_avg = remit_total / 6
            remit_volatility = rem_df["amount_nrs"].std() / (rem_df["amount_nrs"].mean() + 1e-5)

        if "sender_country_code" in rem_df.columns:
            remit_countries = rem_df["sender_country_code"].nunique()

        if "name_match_score" in rem_df.columns:
            remit_min_name_match = rem_df["name_match_score"].min()

        if "disbursement_mode" in rem_df.columns:
            remit_pct_digital = (
                rem_df["disbursement_mode"].isin(["bank_deposit", "mobile_wallet"]).mean()
            )
        has_remit = 1

    feat.update({
        "remit_total_amount_nrs": remit_total,
        "remit_mean_amount_nrs": remit_mean,
        "remit_count": remit_count,
        "remit_num_unique_countries": remit_countries,
        "remit_min_name_match_score": remit_min_name_match,
        "remit_pct_digital": remit_pct_digital,
        "remit_income_volatility": remit_volatility,
        "remit_monthly_avg_inflow": remit_monthly_avg,
        "has_remittance_history": has_remit,
    })

    # ── Utility aggregates ─────────────────────────────────────────────────────
    util_tx_count = 0
    util_avg_bill = 0.0
    util_avg_days_late = -1.0
    util_late_rate = -1.0
    util_early_rate = -1.0
    util_severe_late = 0
    util_digital_rate = -1.0
    util_arrears_ratio = 0.0
    has_util = 0

    if utilities:
        util_df = pd.DataFrame(utilities)
        if "_noise_negative_bill" in util_df.columns:
            util_df = util_df[~util_df["_noise_negative_bill"].fillna(False)]

        util_tx_count = len(util_df)
        has_util = 1

        if "bill_amount_nrs" in util_df.columns:
            bills = pd.to_numeric(util_df["bill_amount_nrs"], errors="coerce").abs()
            util_avg_bill = bills.mean()

        if "days_late" in util_df.columns:
            days = pd.to_numeric(util_df["days_late"], errors="coerce")
            util_avg_days_late = days.mean() if not days.isna().all() else -1.0
            util_late_rate = (days > 0).mean()
            util_early_rate = (days < 0).mean()
            util_severe_late = int((days > 30).sum())

        if "payment_method" in util_df.columns:
            util_digital_rate = util_df["payment_method"].isin(["esewa", "khalti"]).mean()

        if "outstanding_arrears_nrs" in util_df.columns and util_avg_bill > 0:
            avg_arrears = pd.to_numeric(util_df["outstanding_arrears_nrs"], errors="coerce").mean()
            util_arrears_ratio = avg_arrears / (util_avg_bill + 1e-5)

    feat.update({
        "util_tx_count": util_tx_count,
        "util_avg_bill_nrs": util_avg_bill,
        "util_avg_days_late": util_avg_days_late,
        "util_late_payment_rate": util_late_rate,
        "util_early_payment_rate": util_early_rate,
        "util_severe_late_count": util_severe_late,
        "util_digital_payment_rate": util_digital_rate,
        "util_arrears_to_bill_ratio": util_arrears_ratio,
        "has_utility_history": has_util,
    })

    # ── Cooperative features ──────────────────────────────────────────────────
    share_count = 0.0
    total_share_value = 0.0
    last_dividend = 0.0
    outstanding_loan = 0.0
    membership_years = 0.0
    num_sales = 0
    total_sales = 0.0
    has_coop_sales = 0
    coop_repay_status = "none"
    membership_status = "inactive"

    if coop_member:
        share_count = float(coop_member.get("share_count", 0) or 0)
        total_share_value = float(coop_member.get("total_share_value_nrs", 0) or 0)
        last_dividend = float(coop_member.get("last_annual_dividend_nrs", 0) or 0)
        outstanding_loan = float(coop_member.get("outstanding_loan_nrs", 0) or 0)
        membership_year = int(coop_member.get("membership_year_bs", 2080) or 2080)
        membership_years = max(2081 - membership_year, 0)
        coop_repay_status = str(coop_member.get("coop_loan_repayment_status", "none") or "none")
        membership_status = str(coop_member.get("membership_status", "inactive") or "inactive")

    if coop_sales:
        sales_df = pd.DataFrame(coop_sales)
        num_sales = len(sales_df)
        total_sales = sales_df["total_amount_nrs"].sum() if "total_amount_nrs" in sales_df.columns else 0.0
        has_coop_sales = 1

    feat.update({
        "share_count": share_count,
        "total_share_value_nrs": total_share_value,
        "last_annual_dividend_nrs": last_dividend,
        "outstanding_loan_nrs": outstanding_loan,
        "membership_time_yrs": membership_years,
        "num_sales": num_sales,
        "total_sales_amount_nrs": total_sales,
        "has_coop_sales": has_coop_sales,
        "coop_loan_repayment_status": coop_repay_status,
        "membership_status": membership_status,
    })

    return pd.DataFrame([feat])


def _score_to_band(score: int) -> str:
    if score >= 740:
        return "excellent"
    elif score >= 670:
        return "very_good"
    elif score >= 580:
        return "good"
    elif score >= 450:
        return "fair"
    else:
        return "poor"


def _rule_based_score(feat_df: pd.DataFrame) -> int:
    """
    Deterministic rule-based credit score for when the model isn't loaded.
    Produces a score in the 300-850 range based on income and payment signals.
    """
    row = feat_df.iloc[0]

    base = 500.0

    # Income strength
    income = float(row.get("income_agent_monthly_est", 15000))
    confidence = float(row.get("income_confidence", 0.30))
    base += min(income / 1000, 80) * confidence

    # Utility payment reliability
    on_time = float(row.get("util_early_payment_rate", 0))
    if on_time > 0:
        base += on_time * 60
    late_rate = float(row.get("util_late_payment_rate", 0))
    if late_rate > 0:
        base -= late_rate * 80

    # Remittance history
    if row.get("has_remittance_history", 0):
        base += 30
    remit_vol = float(row.get("remit_income_volatility", 0))
    base -= min(remit_vol * 20, 40)

    # Cooperative membership
    if row.get("membership_time_yrs", 0) > 0:
        base += min(float(row.get("membership_time_yrs", 0)) * 5, 30)
    if str(row.get("coop_loan_repayment_status", "")) == "overdue":
        base -= 60

    # Land collateral
    land = float(row.get("land_area_ropani", 0))
    base += min(land * 5, 40)

    # Existing loan burden
    existing_loans = float(row.get("existing_loan_count", 0))
    base -= existing_loans * 20

    return int(np.clip(base, 300, 850))


def map_feature_names_to_human_readable(feature_list):
    mapping = {
        'cat__gender_female': 'Gender: Female',
        'cat__gender_male': 'Gender: Male',
        'cat__marital_status_Divorced': 'Marital Status: Divorced',
        'cat__marital_status_Married': 'Marital Status: Married',
        'cat__marital_status_Single': 'Marital Status: Single',
        'cat__marital_status_Widowed': 'Marital Status: Widowed',
        'cat__occupation_en_Artisan': 'Occupation: Artisan',
        'cat__occupation_en_Business Owner': 'Occupation: Business Owner',
        'cat__occupation_en_Daily Wage Worker': 'Occupation: Daily Wage Worker',
        'cat__occupation_en_Driver': 'Occupation: Driver',
        'cat__occupation_en_Farmer': 'Occupation: Farmer',
        'cat__occupation_en_Government Employee': 'Occupation: Government Employee',
        'cat__occupation_en_Nurse/Health Worker': 'Occupation: Nurse/Health Worker',
        'cat__occupation_en_Remittance Dependent': 'Occupation: Remittance Dependent',
        'cat__occupation_en_Service Worker': 'Occupation: Service Worker',
        'cat__occupation_en_Small Trader': 'Occupation: Small Trader',
        'cat__occupation_en_Teacher': 'Occupation: Teacher',
        'cat__education_level_Bachelors': 'Education Level: Bachelors',
        'cat__education_level_Illiterate': 'Education Level: Illiterate',
        'cat__education_level_Intermediate': 'Education Level: Intermediate',
        'cat__education_level_Masters': 'Education Level: Masters',
        'cat__education_level_PhD': 'Education Level: PhD',
        'cat__education_level_Primary': 'Education Level: Primary',
        'cat__education_level_SLC': 'Education Level: SLC',
        'cat__education_level_Unknown': 'Education Level: Unknown',
        'cat__primary_bank_Citizens Bank': 'Primary Bank: Citizens Bank',
        'cat__primary_bank_Everest Bank': 'Primary Bank: Everest Bank',
        'cat__primary_bank_Global IME Bank': 'Primary Bank: Global IME Bank',
        'cat__primary_bank_Himalayan Bank': 'Primary Bank: Himalayan Bank',
        'cat__primary_bank_Kumari Bank': 'Primary Bank: Kumari Bank',
        'cat__primary_bank_Machhapuchhre Bank': 'Primary Bank: Machhapuchhre Bank',
        'cat__primary_bank_NIC Asia Bank': 'Primary Bank: NIC Asia Bank',
        'cat__primary_bank_Nabil Bank': 'Primary Bank: Nabil Bank',
        'cat__primary_bank_Nepal Investment Bank': 'Primary Bank: Nepal Investment Bank',
        'cat__primary_bank_Prime Bank': 'Primary Bank: Prime Bank',
        'cat__primary_bank_Sanima Bank': 'Primary Bank: Sanima Bank',
        'cat__primary_bank_Siddhartha Bank': 'Primary Bank: Siddhartha Bank',
        'cat__kyc_tier_Tier 1': 'KYC Tier: Tier 1',
        'cat__kyc_tier_Tier 2': 'KYC Tier: Tier 2',
        'cat__kyc_tier_Tier 3': 'KYC Tier: Tier 3',
        'cat__rural_urban_rural': 'Rural/Urban: Rural',
        'cat__rural_urban_semi_urban': 'Rural/Urban: Semi-Urban',
        'cat__rural_urban_urban': 'Rural/Urban: Urban',
        'cat__loan_purpose_agricultural_input': 'Loan Purpose: Agricultural Input',
        'cat__loan_purpose_business_expansion': 'Loan Purpose: Business Expansion',
        'cat__loan_purpose_education': 'Loan Purpose: Education',
        'cat__loan_purpose_emergency_medical': 'Loan Purpose: Emergency Medical',
        'cat__loan_purpose_household_expenses': 'Loan Purpose: Household Expenses',
        'cat__loan_purpose_microenterprise_startup': 'Loan Purpose: Microenterprise Startup',
        'cat__loan_purpose_personal_loan': 'Loan Purpose: Personal Loan',
        'cat__loan_purpose_small_trade': 'Loan Purpose: Small Trade',
        'cat__loan_purpose_vehicle_purchase': 'Loan Purpose: Vehicle Purchase',
        'cat__coop_loan_repayment_status_bad_debt': 'Coop Loan Repay Status: Bad Debt',
        'cat__coop_loan_repayment_status_current': 'Coop Loan Repay Status: Current',
        'cat__coop_loan_repayment_status_defaulted': 'Coop Loan Repay Status: Defaulted',
        'cat__coop_loan_repayment_status_late': 'Coop Loan Repay Status: Late',
        'cat__coop_loan_repayment_status_missing': 'Coop Loan Repay Status: Missing',
        'cat__membership_status_active': 'Coop Membership Status: Active',
        'cat__membership_status_dormant': 'Coop Membership Status: Dormant',
        'cat__membership_status_resigned': 'Coop Membership Status: Resigned',
        'cat__cooperative_type_agricultural': 'Cooperative Type: Agricultural',
        'cat__cooperative_type_consumer': 'Cooperative Type: Consumer',
        'cat__cooperative_type_multipurpose': 'Cooperative Type: Multipurpose',
        'cat__cooperative_type_savings_credit': 'Cooperative Type: Savings & Credit',

        'household_size': 'Household Size',
        'land_area_ropani': 'Land Area (Ropani)',
        'remittance_receiving': 'Receives Remittance',
        'cooperative_member': 'Is Cooperative Member',
        'age': 'Age',
        'requested_amount_nrs': 'Requested Loan Amount (NRS)',
        'requested_tenure_months': 'Requested Loan Tenure (Months)',
        'collateral_value_nrs': 'Collateral Value (NRS)',
        'existing_loan_count': 'Existing Loan Count',
        'credit_bureau_score': 'Credit Bureau Score',
        'doc_completeness_score': 'Document Completeness Score',
        'income_agent_monthly_est': 'Estimated Monthly Income',
        'income_confidence': 'Income Confidence Score',
        'has_cib_history': 'Has Credit Bureau History',
        'no_of_flags': 'Number of Compliance Flags',
        'has_wallet': 'Has Digital Wallet',
        'share_count': 'Cooperative Share Count',
        'total_share_value_nrs': 'Total Cooperative Share Value (NRS)',
        'last_annual_dividend_nrs': 'Last Annual Dividend (NRS)',
        'outstanding_loan_nrs': 'Outstanding Cooperative Loan (NRS)',
        'membership_time_yrs': 'Cooperative Membership Time (Years)',
        'num_sales': 'Cooperative Sales Count',
        'total_sales_amount_nrs': 'Total Cooperative Sales Amount (NRS)',
        'last_sale_years_ago': 'Years Since Last Cooperative Sale',
        'has_coop_sales': 'Has Cooperative Sales',
        'remit_total_amount_nrs': 'Remit Total Amount (NRS)',
        'remit_mean_amount_nrs': 'Remit Mean Amount (NRS)',
        'remit_median_amount_nrs': 'Remit Median Amount (NRS)',
        'remit_max_amount_nrs': 'Remit Max Amount (NRS)',
        'remit_count': 'Remit Count',
        'remit_num_unique_countries': 'Remit Unique Countries',
        'remit_min_name_match_score': 'Remit Min Name Match Score',
        'remit_mean_name_match_score': 'Remit Mean Name Match Score',
        'remit_pct_digital': 'Remit % Digital',
        'remit_std_amount_nrs': 'Remit Std Dev Amount (NRS)',
        'remit_income_volatility': 'Remit Income Volatility',
        'remit_monthly_avg_inflow': 'Remit Monthly Avg Inflow (NRS)',
        'has_remittance_history': 'Has Remittance History',
        'wallet_tx_count': 'Wallet Transaction Count',
        'wallet_total_credit': 'Wallet Total Credit',
        'wallet_total_debit': 'Wallet Total Debit',
        'wallet_topup_total': 'Wallet Top-up Total',
        'wallet_p2p_income_total': 'Wallet P2P Income Total',
        'wallet_utility_spend_total': 'Wallet Utility Spend Total',
        'wallet_agri_spend_total': 'Wallet Agri Spend Total',
        'wallet_loan_repay_total': 'Wallet Loan Repay Total',
        'wallet_withdrawal_total': 'Wallet Withdrawal Total',
        'wallet_midnight_tx_count': 'Wallet Midnight Tx Count',
        'wallet_monthly_avg_income': 'Wallet Monthly Avg Income',
        'wallet_monthly_avg_utility': 'Wallet Monthly Avg Utility',
        'wallet_monthly_avg_withdrawal': 'Wallet Monthly Avg Withdrawal',
        'wallet_agri_reinvestment_rate': 'Wallet Agri Reinvestment Rate',
        'wallet_cash_retention_rate': 'Wallet Cash Retention Rate',
        'util_tx_count': 'Utility Tx Count',
        'util_avg_bill_nrs': 'Utility Avg Bill (NRS)',
        'util_max_bill_nrs': 'Utility Max Bill (NRS)',
        'util_max_arrears_nrs': 'Utility Max Arrears (NRS)',
        'util_avg_arrears_nrs': 'Utility Avg Arrears (NRS)',
        'util_avg_days_late': 'Utility Avg Days Late',
        'util_max_days_late': 'Utility Max Days Late',
        'util_late_payment_rate': 'Utility Late Payment Rate',
        'util_early_payment_rate': 'Utility Early Payment Rate',
        'util_severe_late_count': 'Utility Severe Late Count',
        'util_digital_payment_rate': 'Utility Digital Payment Rate',
        'util_arrears_to_bill_ratio': 'Utility Arrears to Bill Ratio',
        'util_bill_std': 'Utility Bill Std Dev',
        'util_bill_volatility': 'Utility Bill Volatility',
        'has_utility_history': 'Has Utility History'
    }
    return [mapping.get(feature, feature) for feature in feature_list]


def _generate_shap(feat_df: pd.DataFrame, model) -> dict:
    """
    Generate SHAP-style top-10 feature contributions for auditability.
    Supports both raw XGBoost models and scikit-learn Pipelines.
    Falls back to showing raw feature values if shap not available.
    """
    try:
        import shap
        
        # Check if the model is a scikit-learn Pipeline
        if hasattr(model, 'named_steps') and 'preprocess' in model.named_steps and 'model' in model.named_steps:
            preprocessor = model.named_steps['preprocess']
            xgboost_model = model.named_steps['model']
            
            categorical_cols_in_X = [col for col in _CAT_SCORE_FEATURES if col in feat_df.columns]
            numeric_and_bool_cols = [col for col in feat_df.columns if col not in categorical_cols_in_X]
            
            if categorical_cols_in_X and 'cat' in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_['cat']
                cat_feature_names = ohe.get_feature_names_out(categorical_cols_in_X)
            else:
                cat_feature_names = []
                
            feature_names_for_shap = list(cat_feature_names) + list(numeric_and_bool_cols)
            
            # Transform using the pipeline's preprocessor
            X_processed = preprocessor.transform(feat_df)
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names_for_shap).apply(pd.to_numeric, errors='coerce').fillna(0)
            X_processed_df.columns = map_feature_names_to_human_readable(X_processed_df.columns)
            
            explainer = shap.TreeExplainer(xgboost_model)
            shap_values = explainer.shap_values(X_processed_df)
            
            row_shap = shap_values[0]
            features = X_processed_df.columns.tolist()
            pairs = sorted(zip(features, row_shap), key=lambda x: abs(x[1]), reverse=True)
            return {k: round(float(v), 4) for k, v in pairs[:10]}
            
        else:
            # Fallback for plain XGBoost models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feat_df)
            row_shap = shap_values[0]
            features = map_feature_names_to_human_readable(feat_df.columns.tolist())
            pairs = sorted(zip(features, row_shap), key=lambda x: abs(x[1]), reverse=True)
            return {k: round(float(v), 4) for k, v in pairs[:10]}
            
    except Exception as e:
        print(f"[ScoreAgent] SHAP error: {e}")
        # Return raw feature importance proxy
        features = map_feature_names_to_human_readable(feat_df.columns.tolist())
        return {features[i]: round(float(feat_df.iloc[0][col]), 4)
                for i, col in enumerate(feat_df.columns[:10])
                if pd.api.types.is_numeric_dtype(feat_df[col])}


# ── LangGraph Node ────────────────────────────────────────────────────────────

def score_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for the Credit Score Agent.
    Reads income outputs from the clipboard, runs the XGBoost model,
    and writes credit_score, score_band and shap_explanation back.
    """
    global _score_model
    print(f"\n[ScoreAgent] Processing applicant: {state.get('applicant_id')}")
    _load_score_model()

    try:
        feat_df = _build_score_features(state)

        if _score_model is not None:
            # ── ML path ─────────────────────────────────────────────────────
            # Get feature names the model was trained on
            try:
                model_features = _score_model.feature_names_in_
            except AttributeError:
                model_features = _SCORE_FEATURES

            # Align DataFrame columns to match training order
            for col in model_features:
                if col not in feat_df.columns:
                    feat_df[col] = 0

            # DO NOT cast to 'category' here since _score_model is a sklearn Pipeline 
            # with its own OneHotEncoder which expects standard strings/objects.
            # cat_cols = [c for c in _CAT_SCORE_FEATURES if c in feat_df.columns]
            # for col in cat_cols:
            #     feat_df[col] = feat_df[col].astype("category")

            try:
                X = feat_df[model_features]
                raw_pred = _score_model.predict(X)[0]
                credit_score = int(np.clip(round(raw_pred), 300, 850))
                shap_exp = _generate_shap(X, _score_model)
                print(f"[ScoreAgent] ML score: {credit_score}")
            except Exception as ml_exc:
                print(f"[ScoreAgent] ML prediction failed ({ml_exc}), falling back to rule-based.")
                _score_model = None  # Force fallback
        
        if _score_model is None:
            # ── Rule-based fallback ──────────────────────────────────────────
            credit_score = _rule_based_score(feat_df)
            
            # Generate a pseudo-SHAP explanation based on the rule-based logic
            row = feat_df.iloc[0]
            income = float(row.get("income_agent_monthly_est", 15000))
            confidence = float(row.get("income_confidence", 0.30))
            on_time = float(row.get("util_early_payment_rate", 0))
            late_rate = float(row.get("util_late_payment_rate", 0))
            remit_vol = float(row.get("remit_income_volatility", 0))
            existing_loans = float(row.get("existing_loan_count", 0))
            
            shap_exp = {
                "Income Strength": min(income / 1000, 80) * confidence,
                "Utility On-Time Rate": on_time * 60,
                "Utility Late Rate": -(late_rate * 80),
                "Remittance Volatility": -min(remit_vol * 20, 40),
                "Existing Loan Burden": -(existing_loans * 20),
            }
            # Remove 0 impact features
            shap_exp = {k: round(v, 2) for k, v in shap_exp.items() if v != 0}
            
            print(f"[ScoreAgent] Rule-based score: {credit_score}")

        band = _score_to_band(credit_score)

        return {
            "credit_score": credit_score,
            "score_band": band,
            "shap_explanation": shap_exp,
        }

    except Exception as exc:
        print(f"[ScoreAgent] ERROR: {exc}")
        return {
            "credit_score": 400,
            "score_band": "poor",
            "shap_explanation": {},
            "error_message": f"ScoreAgent: {exc}",
        }

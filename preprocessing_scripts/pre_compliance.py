"""
===========================================================================
GIBL Credit Data Validation & Flagging Pipeline (Scale-Independent & Optimized)
===========================================================================
"""

import re
import json
import difflib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ==========================================================================
# Configuration & Static Reference Databases
# ==========================================================================

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"

FILE_PATHS = {
    "applicant_profiles": DATASET_DIR / "applicant_profiles.csv",
    "loan_applications": DATASET_DIR / "loan_applications.csv",
    "mobile_money_transactions": DATASET_DIR / "mobile_money_transactions.parquet",
    "remittance_records": DATASET_DIR / "remittance_records.parquet",
    "utility_payments": DATASET_DIR / "utility_payments.parquet",
    "cooperative_members": DATASET_DIR / "cooperative_members.csv",
    "cooperative_sales": DATASET_DIR / "cooperative_sales.csv",
    "document_registry": DATASET_DIR / "document_registry.csv",
}

# 1. GEOGRAPHY REFERENCE DATABASE (Nested Dict)
# Used to validate province -> district -> municipality hierarchies safely
# without using dynamic dataset size calculations. Customize or load via JSON.
STATIC_GEOGRAPHY_DB = {
    "Koshi": {
        "Jhapa": ["Bhadrapur Municipality", "Damak Municipality", "Birtamod Municipality"],
        "Morang": ["Biratnagar Metropolitan City", "Urlabari Municipality"],
    },
    "Madhesh": {
        "Dhanusha": ["Janakpur Sub-Metropolitan City"],
        "Parsa": ["Birgunj Metropolitan City"],
    },
    "Bagmati": {
        "Kathmandu": ["Kathmandu Metropolitan City", "Kirtipur Municipality", "Budhanilkantha Municipality"],
        "Lalitpur": ["Lalitpur Metropolitan City", "Godawari Municipality"],
        "Bhaktapur": ["Bhaktapur Municipality", "Madhyapur Thimi Municipality"],
        "Kavrepalanchok": ["Banepa Municipality", "Dhulikhel Municipality", "Panauti Municipality"],
    },
    "Gandaki": {
        "Kaski": ["Pokhara Metropolitan City"],
        "Tanahu": ["Vyas Municipality"],
    },
    "Lumbini": {
        "Rupandehi": ["Butwal Sub-Metropolitan City", "Siddharthanagar Municipality"],
        "Banke": ["Nepalgunj Sub-Metropolitan City"],
    },
    "Karnali": {
        "Surkhet": ["Birendranagar Municipality"],
    },
    "Sudurpashchim": {
        "Kailali": ["Dhangadhi Sub-Metropolitan City"],
    }
}

# 2. FX MEDIAN VALUES
# Reference rates to ensure 10x multiplier checks run scale-independently.
STATIC_FX_MEDIANS = {
    "USD": 133.5,
    "INR": 1.60,
    "QAR": 36.6,
    "SAR": 35.5,
    "AED": 36.3,
    "MYR": 28.5,
    "KRW": 0.10,
    "JPY": 0.85
}

# Business thresholds (NRs) -----------------------------------------------
COLLATERAL_REQUIRED_ABOVE_NRS = 500_000       # NRB-COL-005
ENHANCED_KYC_REQUIRED_ABOVE_NRS = 1_000_000   # "PAN-equivalent" proxy threshold
MAX_LTV_RATIO = 0.70                           # Limit for Loan-To-Value (70%)

# NRB interest rate corridor (agricultural / general corridor used in data)
INTEREST_RATE_MIN_PCT = 10.0
INTEREST_RATE_MAX_PCT = 15.0

# Valid enums (from data dictionary) --------------------------------------
VALID_PROVINCES = {
    "Koshi", "Madhesh", "Bagmati", "Gandaki",
    "Lumbini", "Karnali", "Sudurpashchim",
}

VALID_GENDER = {"male", "female"}
VALID_MARITAL_STATUS = {"Married", "Single", "Widowed", "Divorced"}
VALID_EDUCATION = {"Illiterate", "Primary", "SLC", "Intermediate", "Bachelors", "Masters", "PhD"}
VALID_OCCUPATION = {
    "Farmer", "Daily Wage Worker", "Small Trader", "Service Worker",
    "Remittance Dependent", "Artisan", "Government Employee",
    "Business Owner", "Teacher", "Driver", "Nurse/Health Worker",
}
VALID_RURAL_URBAN = {"rural", "semi_urban", "urban"}
VALID_KYC_TIER = {"basic", "mid", "full"}
VALID_COOP_TYPE = {"agricultural", "dairy", "savings_credit", "vegetable", "coffee_tea"}
VALID_DOCUMENT_TYPES = {
    "citizenship_certificate", "utility_bill", "kyc_form",
    "lalpurja", "cooperative_passbook", "remittance_receipt",
}

CITIZENSHIP_REGEX = re.compile(r"^\d{3}-0\d{2}-\d{5}$")
APPLICANT_ID_REGEX = re.compile(r"^AP-\d{6}$")

DOB_MIN_YEAR = 1924
DOB_MAX_DATE = pd.Timestamp.today().normalize()

# Conservative fallback defaults for `flag_to_mail_and_defaulting` --------
DEFAULT_VALUES = {
    "education_level": "Illiterate",
    "marital_status": "Single",
    "occupation_en": "Daily Wage Worker",
    "doc_completeness_score": 0.5,
}

NAME_MATCH_REVIEW_THRESHOLD = 0.80
MAX_EXAMPLE_IDS = 5


# ==========================================================================
# Audit trail infrastructure
# ==========================================================================

def new_audit_log():
    return []


def log_rule(audit_log, table, flag_id, category, field, description,
             mask, df, id_col, policy):
    """
    Record ONE fired validation rule as a single audit-log row with calculated metrics.
    """
    count = int(mask.sum()) if hasattr(mask, "sum") else int(mask)
    total = len(df)
    pct = round(100 * count / total, 3) if total else 0.0

    if count > 0 and id_col in df.columns:
        examples = df.loc[mask, id_col].astype(str).head(MAX_EXAMPLE_IDS).tolist()
    else:
        examples = []

    audit_log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "table": table,
        "flag_id": flag_id,
        "category": category,
        "field": field,
        "description": description,
        "policy_action": policy,
        "affected_rows": count,
        "total_rows": total,
        "affected_pct": pct,
        "example_ids": "; ".join(examples),
    })

    return count


def audit_log_to_dataframe(audit_log):
    if not audit_log:
        return pd.DataFrame(columns=[
            "timestamp", "table", "flag_id", "category", "field",
            "description", "policy_action", "affected_rows", "total_rows",
            "affected_pct", "example_ids",
        ])

    return pd.DataFrame(audit_log).sort_values(
        ["table", "affected_rows"], ascending=[True, False]
    ).reset_index(drop=True)


# ==========================================================================
# Generic helpers
# ==========================================================================

def ensure_flag_columns(df):
    for col in (
        "flag_to_mail_proceed",
        "flag_to_mail_and_wait",
        "flag_to_mail_and_defaulting",
        "flag_auto_corrected",
        "flag_data_quality",
        "flag_manual_review",
        "flag_compliance_block",
    ):
        if col not in df.columns:
            df[col] = ""

    if "hold_for_manual_review" not in df.columns:
        df["hold_for_manual_review"] = False

    return df


def append_flag(df, mask, column, message):
    if not mask.any():
        return df

    existing = df[column]
    separator = np.where(existing.eq(""), "", " | ")
    df.loc[mask, column] = existing[mask] + separator[mask] + message

    if column == "flag_to_mail_and_wait":
        df.loc[mask, "hold_for_manual_review"] = True

    return df


def safe_load(name):
    path = FILE_PATHS[name]
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        print(f"Loaded {name:<28} rows={len(df):>10,}  cols={len(df.columns)}   ({path})")
        return df
    except FileNotFoundError:
        print(f"WARNING: {name} not found at {path} -- skipping table.")
        return None


def coerce_numeric_with_noise_strings(series):
    cleaned = (
        series.astype(str)
        .str.replace("Rs.", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def fast_fuzzy_ratio(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    a_str = str(a).strip().lower()
    b_str = str(b).strip().lower()
    if a_s := (a_str == b_str):
        return 1.0
    return difflib.SequenceMatcher(None, a_str, b_str).ratio()


def vectorised_fuzzy_match(s1, s2):
    ratios = []
    for a, b in zip(s1, s2):
        if pd.isna(a) or pd.isna(b):
            ratios.append(np.nan)
        else:
            a_s = str(a).strip().lower()
            b_s = str(b).strip().lower()
            if a_s == b_s:
                ratios.append(1.0)
            else:
                ratios.append(difflib.SequenceMatcher(None, a_s, b_s).ratio())
    return pd.Series(ratios, index=s1.index)


# ==========================================================================
# SECTION 1 -- APPLICANT PROFILES
# ==========================================================================

def check_applicant_profile_basics(df, audit_log):
    print("\n" + "=" * 80)
    print("APPLICANT PROFILES -- CHECKS")
    print("=" * 80)

    dup_mask = df.duplicated(subset=["applicant_id"], keep="first")
    log_rule(
        audit_log, "applicant_profiles", "AP-DUP-001", "flag_data_quality",
        "applicant_id", "Duplicate applicant_id detected.",
        dup_mask, df, "applicant_id", "Row retained; flagged for auditor review.",
    )
    df = append_flag(df, dup_mask, "flag_data_quality", "duplicate_applicant_id")

    id_mask = ~df["applicant_id"].astype(str).str.match(APPLICANT_ID_REGEX)
    log_rule(
        audit_log, "applicant_profiles", "AP-ID-001", "flag_data_quality",
        "applicant_id", "applicant_id violates standard formats.",
        id_mask, df, "applicant_id", "Flagged for standard schema resolution.",
    )
    df = append_flag(df, id_mask, "flag_data_quality", "malformed_applicant_id")

    return df


def check_names(df, audit_log):
    for col in ("full_name_en", "father_name_en", "grandfather_name_en"):
        if col not in df.columns:
            continue

        is_text = df[col].notna() & (df[col].astype(str).str.len() > 0)
        all_caps = is_text & df[col].astype(str).str.isupper()
        all_lower = is_text & df[col].astype(str).str.islower()
        format_mask = all_caps | all_lower

        log_rule(
            audit_log, "applicant_profiles", f"AP-NAME-FMT-{col}", "flag_auto_corrected",
            col, f"{col} has incorrect casing; title case applied.",
            format_mask, df, "applicant_id", "Auto-corrected in place.",
        )
        df.loc[format_mask, col] = df.loc[format_mask, col].astype(str).str.title()
        df = append_flag(df, format_mask, "flag_auto_corrected", f"{col}_case_normalised")

        if col in ("father_name_en", "grandfather_name_en"):
            missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            log_rule(
                audit_log, "applicant_profiles", f"AP-NAME-MISS-{col}", "flag_to_mail_and_wait",
                col, f"{col} missing; crucial for collateral/document validation.",
                missing_mask, df, "applicant_id", "Held for documentation.",
            )
            df = append_flag(df, missing_mask, "flag_to_mail_and_wait", f"missing_{col}")

    return df


def check_citizenship(df, audit_log):
    malformed_mask = ~df["citizenship_number"].astype(str).str.match(CITIZENSHIP_REGEX)
    log_rule(
        audit_log, "applicant_profiles", "AP-CIT-001", "flag_to_mail_and_wait",
        "citizenship_number", "Citizenship identification fails regex validation.",
        malformed_mask, df, "applicant_id", "Application placed on hold.",
    )
    df = append_flag(df, malformed_mask, "flag_to_mail_and_wait", "malformed_citizenship_number")
    return df


def check_dob(df, audit_log):
    dob = pd.to_datetime(df["dob_ad"], errors="coerce")
    invalid_parse = dob.isna() & df["dob_ad"].notna()
    future_dob = dob.notna() & (dob > DOB_MAX_DATE)
    too_old_dob = dob.notna() & (dob.dt.year < DOB_MIN_YEAR)
    bad_dob_mask = invalid_parse | future_dob | too_old_dob

    log_rule(
        audit_log, "applicant_profiles", "AP-DOB-001", "flag_to_mail_and_wait",
        "dob_ad", "Legal DOB is unparsable or out of valid historical range.",
        bad_dob_mask, df, "applicant_id", "Held; verification requested.",
    )
    df = append_flag(df, bad_dob_mask, "flag_to_mail_and_wait", "invalid_dob")
    return df


def check_compliance_pep_fraud_profiles(df, audit_log):
    """
    Check for PEP Status or Fraud DB indicators at the profile stage.
    """
    if "pep_flag" in df.columns:
        pep_mask = df["pep_flag"] == True
        log_rule(
            audit_log, "applicant_profiles", "AP-PEP-001", "flag_manual_review",
            "pep_flag", "Applicant is identified as a Politically Exposed Person (PEP).",
            pep_mask, df, "applicant_id", "Placed on immediate manual review hold."
        )
        df = append_flag(df, pep_mask, "flag_manual_review", "politically_exposed_person_flag")
        df.loc[pep_mask, "hold_for_manual_review"] = True

    if "fraud_db_flag" in df.columns:
        fraud_mask = df["fraud_db_flag"] == True
        log_rule(
            audit_log, "applicant_profiles", "AP-FRD-001", "flag_manual_review",
            "fraud_db_flag", "Applicant exists in external Fraud Database registry.",
            fraud_mask, df, "applicant_id", "Placed on immediate manual review hold."
        )
        df = append_flag(df, fraud_mask, "flag_manual_review", "fraud_registry_match")
        df.loc[fraud_mask, "hold_for_manual_review"] = True

    return df


def check_geography(df, audit_log):
    """
    Scale-independent Geography Check. Performs vectorised mapping against the static
    lookups from STATIC_GEOGRAPHY_DB instead of calculating dynamic batch statistics.
    """
    missing_geo = (
        df["province_en"].isna() | df["district_en"].isna() | df["municipality_en"].isna()
    )
    log_rule(
        audit_log, "applicant_profiles", "AP-GEO-MISSING", "flag_to_mail_proceed",
        "province_en/district_en/municipality_en", "Missing core structural address components.",
        missing_geo, df, "applicant_id", "Low risk feature; proceed and email.",
    )
    df = append_flag(df, missing_geo, "flag_to_mail_proceed", "missing_address_component")

    invalid_province = df["province_en"].notna() & ~df["province_en"].isin(VALID_PROVINCES)
    log_rule(
        audit_log, "applicant_profiles", "AP-GEO-PROV", "flag_to_mail_proceed",
        "province_en", "Province is not part of the 7 valid administrative regions.",
        invalid_province, df, "applicant_id", "Proceed and seek clarification.",
    )
    df = append_flag(df, invalid_province, "flag_to_mail_proceed", "invalid_province")

    # Map static references to verify compatibility
    valid_province_districts = set()
    valid_district_municipalities = set()
    for prov, dists in STATIC_GEOGRAPHY_DB.items():
        for dist, munis in dists.items():
            valid_province_districts.add((prov, dist))
            for muni in munis:
                valid_district_municipalities.add((dist, muni))

    pd_combo = df["province_en"].fillna("") + "||" + df["district_en"].fillna("")
    valid_pd_strings = {f"{p}||{d}" for p, d in valid_province_districts}
    pd_ok = pd_combo.isin(valid_pd_strings) | df["province_en"].isna() | df["district_en"].isna()

    dm_combo = df["district_en"].fillna("") + "||" + df["municipality_en"].fillna("")
    valid_dm_strings = {f"{d}||{m}" for d, m in valid_district_municipalities}
    dm_ok = dm_combo.isin(valid_dm_strings) | df["district_en"].isna() | df["municipality_en"].isna()

    geo_combo_mask = ~(pd_ok & dm_ok)
    log_rule(
        audit_log, "applicant_profiles", "AP-GEO-COMBO", "flag_to_mail_proceed",
        "province_en/district_en/municipality_en", "Geographical mismatch verified against reference database.",
        geo_combo_mask, df, "applicant_id", "Proceed and queue for validation.",
    )
    df = append_flag(df, geo_combo_mask, "flag_to_mail_proceed", "geography_combination_mismatch")

    return df


def check_gender_marital_occupation(df, audit_log):
    bad_gender = ~df["gender"].isin(VALID_GENDER) & df["gender"].notna()
    df = append_flag(df, bad_gender, "flag_data_quality", "invalid_gender")

    marital_mask = df["marital_status"].isna() | ~df["marital_status"].isin(VALID_MARITAL_STATUS)
    df.loc[marital_mask, "marital_status"] = DEFAULT_VALUES["marital_status"]
    df = append_flag(df, marital_mask, "flag_to_mail_and_defaulting", f"marital_status_defaulted_to_{DEFAULT_VALUES['marital_status']}")

    occ_mask = df["occupation_en"].isna() | ~df["occupation_en"].isin(VALID_OCCUPATION)
    df.loc[occ_mask, "occupation_en"] = DEFAULT_VALUES["occupation_en"]
    df = append_flag(df, occ_mask, "flag_to_mail_and_defaulting", f"occupation_defaulted_to_{DEFAULT_VALUES['occupation_en']}")

    return df


def check_ward_and_phone(df, audit_log):
    ward_mask = df["ward_no"].isna() | (df["ward_no"] == 0)
    df = append_flag(df, ward_mask, "flag_to_mail_proceed", "missing_or_invalid_ward")
    df.loc[df["ward_no"] == 0, "ward_no"] = np.nan

    phone = df["phone_primary"].astype(str)
    phone_mask = df["phone_primary"].isna() | ~phone.str.fullmatch(r"\d{10}")
    df = append_flag(df, phone_mask, "flag_to_mail_proceed", "missing_or_invalid_phone")
    return df


def check_education_level(df, audit_log):
    edu_mask = df["education_level"].isna() | ~df["education_level"].isin(VALID_EDUCATION)
    df.loc[edu_mask, "education_level"] = DEFAULT_VALUES["education_level"]
    df = append_flag(df, edu_mask, "flag_to_mail_and_defaulting", f"education_defaulted_to_{DEFAULT_VALUES['education_level']}")
    return df


def check_household_and_land(df, audit_log):
    hh_mask = df["household_size"].notna() & ~df["household_size"].between(1, 8)
    df = append_flag(df, hh_mask, "flag_data_quality", "household_size_out_of_range")

    land_mask = df["land_area_ropani"].notna() & ~df["land_area_ropani"].between(0, 21)
    df = append_flag(df, land_mask, "flag_data_quality", "land_area_out_of_range")
    return df


def check_wallet_consistency(df, audit_log):
    esewa_mask = df["has_esewa_account"] & df["esewa_account_id"].isna()
    df = append_flag(df, esewa_mask, "flag_data_quality", "esewa_id_missing_despite_active_flag")

    khalti_mask = df["has_khalti_account"] & df["khalti_account_id"].isna()
    df = append_flag(df, khalti_mask, "flag_data_quality", "khalti_id_missing_despite_active_flag")
    return df


def run_applicant_profile_checks(df, audit_log):
    df = ensure_flag_columns(df)
    df = check_applicant_profile_basics(df, audit_log)
    df = check_names(df, audit_log)
    df = check_citizenship(df, audit_log)
    df = check_dob(df, audit_log)
    df = check_compliance_pep_fraud_profiles(df, audit_log)
    df = check_geography(df, audit_log)
    df = check_gender_marital_occupation(df, audit_log)
    df = check_ward_and_phone(df, audit_log)
    df = check_education_level(df, audit_log)
    df = check_household_and_land(df, audit_log)
    df = check_wallet_consistency(df, audit_log)
    return df


# ==========================================================================
# SECTION 2 -- LOAN APPLICATIONS
# ==========================================================================

def check_loan_amount_fields(df, audit_log):
    df["requested_amount_clean"] = coerce_numeric_with_noise_strings(df["requested_amount_nrs"])

    was_string_or_negative = (
        df["requested_amount_nrs"].astype(str).str.contains(r"[A-Za-z,]", regex=True)
        | (df["requested_amount_clean"] < 0)
    )
    log_rule(
        audit_log, "loan_applications", "LA-AMT-001", "flag_auto_corrected",
        "requested_amount_nrs", "Amount field contained format errors or negative metrics.",
        was_string_or_negative, df, "application_id", "Auto-coerced to clean positive numeric.",
    )
    df["requested_amount_clean"] = df["requested_amount_clean"].abs()
    df = append_flag(df, was_string_or_negative, "flag_auto_corrected", "requested_amount_sanitized")

    tenure_mask = df["requested_tenure_months"].isna()
    df = append_flag(df, tenure_mask, "flag_to_mail_proceed", "missing_tenure_specification")
    return df


def check_compliance_blocks_and_holds(df, audit_log):
    """
    Enforces immediate decision logic:
    - NRB Defaulter/Blacklist status -> Immediate Reject
    - AML, PEP, or Fraud Match -> Manual Review Hold
    """
    # 1. Immediate Hard Rejection on Blacklist Defaulters
    blacklist_mask = df["nrb_blacklist_flag"] == True
    log_rule(
        audit_log, "loan_applications", "LA-COMP-BLACK", "flag_compliance_block",
        "nrb_blacklist_flag", "Applicant flagged in NRB central credit blacklist registry.",
        blacklist_mask, df, "application_id",
        "Hard Stop. final_decision set immediately to REJECT. Overriding holds.",
    )
    df = append_flag(df, blacklist_mask, "flag_compliance_block", "nrb_blacklist_enforced_reject")
    df.loc[blacklist_mask, "final_decision"] = "reject"
    df.loc[blacklist_mask, "hold_for_manual_review"] = False  # Vetoed. No manual review can approve.

    # 2. AML Flags -> Manual Hold
    aml_mask = df["aml_flag"] == True
    log_rule(
        audit_log, "loan_applications", "LA-COMP-AML", "flag_compliance_block",
        "aml_flag", "AML automated rules raised high-risk activity flags.",
        aml_mask, df, "application_id", "Disbursement blocked; set to hold status.",
    )
    df = append_flag(df, aml_mask, "flag_compliance_block", "aml_flag_triggered")
    df.loc[aml_mask, "hold_for_manual_review"] = True

    # 3. Safe check for PEP and Fraud DB at Loan Level
    if "pep_flag" in df.columns:
        pep_mask = df["pep_flag"] == True
        log_rule(
            audit_log, "loan_applications", "LA-COMP-PEP", "flag_manual_review",
            "pep_flag", "Compliance alert: PEP Match at loan application evaluation.",
            pep_mask, df, "application_id", "Placed on Manual Compliance Hold."
        )
        df = append_flag(df, pep_mask, "flag_manual_review", "pep_compliance_hold")
        df.loc[pep_mask, "hold_for_manual_review"] = True

    if "fraud_db_flag" in df.columns:
        fraud_mask = df["fraud_db_flag"] == True
        log_rule(
            audit_log, "loan_applications", "LA-COMP-FRD", "flag_manual_review",
            "fraud_db_flag", "Security alert: Match found in external fraud list.",
            fraud_mask, df, "application_id", "Placed on Manual Compliance Hold."
        )
        df = append_flag(df, fraud_mask, "flag_manual_review", "fraud_db_compliance_hold")
        df.loc[fraud_mask, "hold_for_manual_review"] = True

    # 4. Interest Rate Corridor Compliance (NRB)
    df["interest_rate_clean"] = pd.to_numeric(df["interest_rate_pct"], errors="coerce")
    corridor_mask = (
        (df["interest_rate_clean"] > 0)
        & ~df["interest_rate_clean"].between(INTEREST_RATE_MIN_PCT, INTEREST_RATE_MAX_PCT)
    )
    log_rule(
        audit_log, "loan_applications", "LA-COMP-INT", "flag_compliance_block",
        "interest_rate_pct", f"Pricing deviates from NRB Corridor ({INTEREST_RATE_MIN_PCT}%-{INTEREST_RATE_MAX_PCT}%).",
        corridor_mask, df, "application_id", "Disbursement blocked; held for rate correction.",
    )
    df = append_flag(df, corridor_mask, "flag_compliance_block", "interest_rate_outside_corridor")

    return df


def check_loan_to_value_adequacy(df, audit_log):
    """
    Validates loan sizes relative to pledged collateral values (LTV Ratio).
    """
    collateral_val = pd.to_numeric(df["collateral_value_nrs"], errors="coerce").fillna(0.0)
    requested_amt = df["requested_amount_clean"]

    # Protect against division by zero
    ltv = np.where(collateral_val > 0.0, requested_amt / collateral_val, 0.0)
    df["calculated_ltv"] = ltv

    ltv_excess_mask = (collateral_val > 0.0) & (ltv > MAX_LTV_RATIO)
    log_rule(
        audit_log, "loan_applications", "LA-VAL-LTV", "flag_manual_review",
        "calculated_ltv", f"LTV Ratio of ({MAX_LTV_RATIO*100}%) exceeded.",
        ltv_excess_mask, df, "application_id", "Under-collateralised application placed on manual review hold.",
    )
    df = append_flag(df, ltv_excess_mask, "flag_manual_review", "ltv_ratio_limit_exceeded")
    df.loc[ltv_excess_mask, "hold_for_manual_review"] = True

    return df


def run_loan_application_checks(df, audit_log):
    print("\n" + "=" * 80)
    print("LOAN APPLICATIONS -- CHECKS")
    print("=" * 80)

    df = ensure_flag_columns(df)
    df = check_loan_amount_fields(df, audit_log)
    df = check_compliance_blocks_and_holds(df, audit_log)
    df = check_loan_to_value_adequacy(df, audit_log)

    return df


# ==========================================================================
# SECTION 3 -- MOBILE MONEY TRANSACTIONS
# ==========================================================================

def check_transaction_amount_and_dates(df, audit_log):
    df["amount_nrs_clean"] = coerce_numeric_with_noise_strings(df["amount_nrs"])
    was_string_mask = df["amount_nrs"].astype(str).str.contains(",", regex=False)
    df = append_flag(df, was_string_mask, "flag_auto_corrected", "amount_formatting_coerced")

    df["transaction_datetime"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    future_mask = df["transaction_datetime"] > pd.Timestamp.now()
    log_rule(
        audit_log, "mobile_money_transactions", "TX-DATE-001", "flag_manual_review",
        "transaction_date", "Transactions dated with future timestamps.",
        future_mask, df, "transaction_id", "Flagged for integrity review.",
    )
    df = append_flag(df, future_mask, "flag_manual_review", "future_timestamp_detected")
    return df


def detect_transaction_fraud_patterns(df, audit_log):
    dt = df["transaction_datetime"]
    is_midnight = dt.dt.hour.eq(0) & dt.dt.minute.eq(0) & dt.dt.second.eq(0)
    is_round_amount = df["amount_nrs_clean"].notna() & (df["amount_nrs_clean"] % 1000 == 0)
    midnight_round_mask = is_midnight & is_round_amount

    df = append_flag(df, midnight_round_mask, "flag_manual_review", "suspected_midnight_scam_pattern")

    # Time binning is safe on 1-row dataframes as count will evaluate as 1 (no exception)
    epoch_seconds = dt.astype("int64") // 10**9
    bin_id = epoch_seconds // 300
    bin_key = df["applicant_id"].astype(str) + "_" + bin_id.astype(str)
    bin_counts = bin_key.map(bin_key.value_counts())
    velocity_mask = bin_counts >= 10

    applicant_mean_amount = df.groupby("applicant_id")["amount_nrs_clean"].transform("mean")
    amount_outlier_mask = (
        (applicant_mean_amount > 0)
        & df["amount_nrs_clean"].notna()
        & (df["amount_nrs_clean"] > 8 * applicant_mean_amount)
    )

    burst_mask = velocity_mask | amount_outlier_mask
    df = append_flag(df, burst_mask, "flag_manual_review", "velocity_burst_mismatch")

    return df


def run_mobile_money_checks(df, audit_log):
    print("\n" + "=" * 80)
    print("MOBILE MONEY TRANSACTIONS -- CHECKS")
    print("=" * 80)

    df = ensure_flag_columns(df)
    df = check_transaction_amount_and_dates(df, audit_log)
    df = detect_transaction_fraud_patterns(df, audit_log)
    return df


# ==========================================================================
# SECTION 4 -- REMITTANCE RECORDS
# ==========================================================================

def check_remittance_amounts(df, audit_log):
    """
    Scale-independent exchange rate checking using a static baseline.
    Avoids dynamic medians which fail on 1-row tests.
    """
    baseline_fx = df["foreign_currency_code"].map(STATIC_FX_MEDIANS).fillna(133.5)
    impossible_rate_mask = df["exchange_rate"] > (10 * baseline_fx)

    log_rule(
        audit_log, "remittance_records", "REM-RATE-001", "flag_manual_review",
        "exchange_rate", "Exchange rate exceeds reference limits (likely 10x decimal error).",
        impossible_rate_mask, df, "remittance_id", "Flagged for correction.",
    )
    df = append_flag(df, impossible_rate_mask, "flag_manual_review", "exchange_rate_outlier")

    usd_but_not_usa = (
        (df["foreign_currency_code"] == "USD")
        & ~df["sender_country_name"].isin(["USA", "United States"])
    )
    df = append_flag(df, usd_but_not_usa, "flag_data_quality", "currency_country_code_mismatch")
    return df


def check_remittance_name_matching(df, profiles_df, audit_log):
    low_match_mask = df["name_match_score"] < NAME_MATCH_REVIEW_THRESHOLD
    df = append_flag(df, low_match_mask, "flag_manual_review", "unreliable_sender_identity_match")

    if profiles_df is not None and "receiver_name_en" in df.columns:
        name_lookup = profiles_df.set_index("applicant_id")["full_name_en"]
        df["_temp_app_name"] = df["applicant_id"].map(name_lookup)

        receiver_ratio = vectorised_fuzzy_match(df["receiver_name_en"], df["_temp_app_name"])
        df.drop(columns=["_temp_app_name"], inplace=True)

        receiver_mismatch_mask = receiver_ratio.notna() & (receiver_ratio < NAME_MATCH_REVIEW_THRESHOLD)
        log_rule(
            audit_log, "remittance_records", "REM-NAME-002", "flag_manual_review",
            "receiver_name_en", "Recipient name deviates significantly from legal customer profile name.",
            receiver_mismatch_mask, df, "remittance_id", "Routing for identification verification.",
        )
        df = append_flag(df, receiver_mismatch_mask, "flag_manual_review", "receiver_name_profile_mismatch")

    return df


def run_remittance_checks(df, profiles_df, audit_log):
    print("\n" + "=" * 80)
    print("REMITTANCE RECORDS -- CHECKS")
    print("=" * 80)

    df = ensure_flag_columns(df)
    df = check_remittance_amounts(df, audit_log)
    df = check_remittance_name_matching(df, profiles_df, audit_log)
    return df


# ==========================================================================
# SECTION 5 -- UTILITY PAYMENTS
# ==========================================================================

def run_utility_payment_checks(df, audit_log):
    print("\n" + "=" * 80)
    print("UTILITY PAYMENTS -- CHECKS")
    print("=" * 80)

    df = ensure_flag_columns(df)
    negative_bill_mask = df["bill_amount_nrs"] < 0
    df.loc[negative_bill_mask, "bill_amount_nrs"] = df.loc[negative_bill_mask, "bill_amount_nrs"].abs()
    df = append_flag(df, negative_bill_mask, "flag_auto_corrected", "negative_utility_amount_corrected")

    return df


# ==========================================================================
# SECTIONS 6 & 7 -- COOPERATIVES
# ==========================================================================

def run_cooperative_member_checks(df, audit_log):
    print("\n" + "=" * 80)
    print("COOPERATIVE MEMBERS -- CHECKS")
    print("=" * 80)
    df = ensure_flag_columns(df)
    return df


def run_cooperative_sales_checks(df, coop_members_df, audit_log):
    print("\n" + "=" * 80)
    print("COOPERATIVE SALES -- CHECKS")
    print("=" * 80)
    df = ensure_flag_columns(df)
    return df


# ==========================================================================
# SECTION 8 -- DOCUMENT REGISTRY
# ==========================================================================

def run_document_registry_checks(df, audit_log):
    print("\n" + "=" * 80)
    print("DOCUMENT REGISTRY -- CHECKS")
    print("=" * 80)
    df = ensure_flag_columns(df)
    return df


# ==========================================================================
# SECTION 9 -- CROSS-TABLE VALIDATION
# ==========================================================================

def cross_check_ghost_membership(profiles, coop_members, audit_log):
    if coop_members is None:
        return profiles

    member_ids = set(coop_members["applicant_id"])
    ghost_mask = profiles["cooperative_member"].fillna(False) & ~profiles["applicant_id"].isin(member_ids)

    log_rule(
        audit_log, "applicant_profiles", "XTAB-GHOST-001", "flag_to_mail_and_wait",
        "cooperative_member", "Profile claims membership, but cooperative database record missing.",
        ghost_mask, profiles, "applicant_id", "Placing on hold; cooperative passbook requested.",
    )
    profiles = append_flag(profiles, ghost_mask, "flag_to_mail_and_wait", "ghost_cooperative_membership")
    return profiles


def cross_check_loan_profile_geo(loans, profiles, audit_log):
    if profiles is None:
        return loans

    profile_geo = profiles.set_index("applicant_id")[["province_en", "district_en", "municipality_en"]]
    joined = loans.join(profile_geo, on="applicant_id", rsuffix="_profile")

    mismatch_mask = (
        (joined["province_en"] != joined["province_en_profile"])
        | (joined["district_en"] != joined["district_en_profile"])
        | (joined["municipality_en"] != joined["municipality_en_profile"])
    ).fillna(False)

    log_rule(
        audit_log, "loan_applications", "XTAB-GEO-001", "flag_data_quality",
        "province_en", "Application-level address contradicts master customer profile address.",
        mismatch_mask, loans, "application_id", "Flagged; profile records assumed as primary truth.",
    )
    loans = append_flag(loans, mismatch_mask, "flag_data_quality", "loan_profile_address_mismatch")
    return loans


def cross_check_required_documents(profiles, loans, documents, audit_log):
    if documents is None or loans is None or profiles is None:
        return profiles, loans

    docs_by_applicant = documents.groupby("applicant_id")["document_type"].apply(set)
    verified_docs_by_applicant = (
        documents[documents["verified_by_agent"] == True]
        .groupby("applicant_id")["document_type"].apply(set)
    )

    base = profiles.set_index("applicant_id")
    loan_by_applicant = loans.set_index("applicant_id")

    has_docs = docs_by_applicant.reindex(base.index, fill_value=set())
    has_verified = verified_docs_by_applicant.reindex(base.index, fill_value=set())

    def _missing(doc_type, series):
        return ~series.apply(lambda s: doc_type in s)

    missing_citizenship = _missing("citizenship_certificate", has_docs)
    profiles = append_flag(
        profiles, profiles["applicant_id"].isin(base.index[missing_citizenship]),
        "flag_to_mail_and_wait", "missing_citizenship_certificate_document",
    )

    needs_full_kyc_docs = base["kyc_tier"].isin(["mid", "full"])
    missing_kyc_form = needs_full_kyc_docs & _missing("kyc_form", has_docs)
    profiles = append_flag(
        profiles, profiles["applicant_id"].isin(base.index[missing_kyc_form]),
        "flag_to_mail_and_wait", "missing_kyc_form_document",
    )

    missing_utility_bill = needs_full_kyc_docs & _missing("utility_bill", has_docs)
    profiles = append_flag(
        profiles, profiles["applicant_id"].isin(base.index[missing_utility_bill]),
        "flag_to_mail_and_wait", "missing_utility_bill_document",
    )

    # Cross-verify loan size compliance requirements
    loan_by_applicant_aligned = loan_by_applicant.reindex(base.index)
    requested = loan_by_applicant_aligned["requested_amount_clean"]
    collateral_type = loan_by_applicant_aligned["collateral_type"]

    needs_collateral = requested > COLLATERAL_REQUIRED_ABOVE_NRS
    has_land_collateral = collateral_type == "land"
    has_lalpurja_doc = has_docs.apply(lambda s: "lalpurja" in s)

    missing_collateral_mask = needs_collateral & (~has_land_collateral | ~has_lalpurja_doc)
    flagged_applicants = set(base.index[missing_collateral_mask.fillna(False)])
    loans = append_flag(
        loans, loans["applicant_id"].isin(flagged_applicants),
        "flag_to_mail_and_wait", "missing_lalpurja_for_large_loan",
    )

    needs_enhanced_kyc = requested > ENHANCED_KYC_REQUIRED_ABOVE_NRS
    has_full_kyc_tier = base["kyc_tier"] == "full"
    has_verified_kyc_form = has_verified.apply(lambda s: "kyc_form" in s)
    has_verified_citizenship = has_verified.apply(lambda s: "citizenship_certificate" in s)

    missing_enhanced_kyc_mask = needs_enhanced_kyc & ~(
        has_full_kyc_tier & has_verified_kyc_form & has_verified_citizenship
    )
    flagged_applicants_2 = set(base.index[missing_enhanced_kyc_mask.fillna(False)])
    loans = append_flag(
        loans, loans["applicant_id"].isin(flagged_applicants_2),
        "flag_to_mail_and_wait", "missing_enhanced_kyc_pan_equivalent_for_large_loan",
    )

    return profiles, loans


def cross_check_doc_completeness_score(loans, documents, audit_log):
    if documents is None:
        return loans

    verified_counts = (
        documents[documents["verified_by_agent"] == True]
        .groupby("applicant_id").size()
    )
    total_counts = documents.groupby("applicant_id").size()
    derived_ratio = (verified_counts / total_counts).reindex(loans["applicant_id"]).values

    missing_mask = loans["doc_completeness_score"].isna()
    derivable_mask = missing_mask & ~pd.isna(derived_ratio)

    loans.loc[derivable_mask, "doc_completeness_score"] = derived_ratio[derivable_mask]
    loans = append_flag(loans, derivable_mask, "flag_auto_corrected", "doc_completeness_score_derived_from_registry")

    still_missing_mask = loans["doc_completeness_score"].isna()
    loans.loc[still_missing_mask, "doc_completeness_score"] = DEFAULT_VALUES["doc_completeness_score"]
    loans = append_flag(loans, still_missing_mask, "flag_to_mail_and_defaulting",
                         f"doc_completeness_score_defaulted_to_{DEFAULT_VALUES['doc_completeness_score']}")

    return loans


def run_cross_table_checks(profiles, loans, transactions, remittances, utilities,
                            coop_members, coop_sales, documents, audit_log):
    print("\n" + "=" * 80)
    print("CROSS-TABLE VALIDATIONS")
    print("=" * 80)

    profiles = cross_check_ghost_membership(profiles, coop_members, audit_log)
    loans = cross_check_loan_profile_geo(loans, profiles, audit_log)
    profiles, loans = cross_check_required_documents(profiles, loans, documents, audit_log)
    loans = cross_check_doc_completeness_score(loans, documents, audit_log)

    return profiles, loans


# ==========================================================================
# SECTION 10 -- OPTIMIZED MERGE / APPLICANT-LEVEL FEATURE TABLE
# ==========================================================================

def _combine_flag_text(*series_list):
    result = None
    for series in series_list:
        series = series.fillna("")
        if result is None:
            result = series
        else:
            sep = np.where((result == "") | (series == ""), "", " | ")
            result = result + sep + series
    return result


def build_merged_dataset(profiles, loans, transactions, remittances, utilities,
                          coop_members, coop_sales, documents, audit_log):
    """
    Combines application and profile datasets. Groupby aggregations have been
    optimized to run in linear-time vectorised passes, stripping away all slow loops.
    """
    print("\n" + "=" * 80)
    print("BUILDING APPLICANT-LEVEL MERGED DATASET")
    print("=" * 80)

    merged = loans.merge(
        profiles, on="applicant_id", how="left", suffixes=("_loan", "_profile"),
    )

    # Fast Vectorised Aggregations (No lambdas accessing parent variables)
    if transactions is not None:
        transactions["_is_credit"] = (transactions["direction"] == "credit").astype(int)
        transactions["_is_debit"] = (transactions["direction"] == "debit").astype(int)
        transactions["_credit_val"] = transactions["amount_nrs_clean"] * transactions["_is_credit"]
        transactions["_debit_val"] = transactions["amount_nrs_clean"] * transactions["_is_debit"]
        transactions["_is_fraud_review"] = transactions["flag_manual_review"].fillna("").ne("").astype(int)

        tx_agg = transactions.groupby("applicant_id").agg(
            tx_count=("transaction_id", "count"),
            tx_credit_total=("_credit_val", "sum"),
            tx_debit_total=("_debit_val", "sum"),
            tx_fraud_review_count=("_is_fraud_review", "sum"),
        ).reset_index()

        merged = merged.merge(tx_agg, on="applicant_id", how="left")
        transactions.drop(columns=["_is_credit", "_is_debit", "_credit_val", "_debit_val", "_is_fraud_review"], inplace=True)

    if remittances is not None:
        remittances["_is_review"] = remittances["flag_manual_review"].fillna("").ne("").astype(int)

        rem_agg = remittances.groupby("applicant_id").agg(
            remittance_total_nrs=("amount_nrs", "sum"),
            remittance_count=("remittance_id", "count"),
            remittance_avg_name_match=("name_match_score", "mean"),
            remittance_review_count=("_is_review", "sum"),
        ).reset_index()

        merged = merged.merge(rem_agg, on="applicant_id", how="left")
        remittances.drop(columns=["_is_review"], inplace=True)

    if utilities is not None:
        util_agg = utilities.groupby("applicant_id").agg(
            utility_avg_on_time_rate=("cumulative_on_time_rate", "mean"),
            utility_total_arrears_nrs=("outstanding_arrears_nrs", "sum"),
            utility_bill_count=("payment_id", "count"),
        ).reset_index()
        merged = merged.merge(util_agg, on="applicant_id", how="left")

    if coop_sales is not None:
        sales_agg = coop_sales.groupby("applicant_id").agg(
            coop_sales_total_nrs=("total_amount_nrs", "sum"),
            coop_sales_count=("sale_id", "count"),
        ).reset_index()
        merged = merged.merge(sales_agg, on="applicant_id", how="left")

    if coop_members is not None:
        member_cols = coop_members[[
            "applicant_id", "cooperative_type", "membership_year_bs",
            "outstanding_loan_nrs", "coop_loan_repayment_status", "membership_status",
        ]]
        merged = merged.merge(member_cols, on="applicant_id", how="left", suffixes=("", "_coop"))

    if documents is not None:
        doc_agg = documents.groupby("applicant_id").agg(
            document_count=("document_id", "count"),
            documents_verified_count=("verified_by_agent", "sum"),
            document_anomaly_count=("anomaly_flag", "sum"),
        ).reset_index()
        merged = merged.merge(doc_agg, on="applicant_id", how="left")

    # Unified Flag Aggregation
    profile_cols = {c: c for c in (
        "flag_to_mail_proceed", "flag_to_mail_and_wait", "flag_to_mail_and_defaulting",
        "flag_manual_review", "flag_data_quality", "flag_compliance_block",
    )}

    for base_col in profile_cols:
        loan_col = f"{base_col}_loan" if f"{base_col}_loan" in merged.columns else base_col
        profile_col = f"{base_col}_profile" if f"{base_col}_profile" in merged.columns else None
        if profile_col:
            merged[f"merged_{base_col}"] = _combine_flag_text(merged[loan_col], merged[profile_col])
        else:
            merged[f"merged_{base_col}"] = merged[loan_col]

    merged["merged_hold_for_manual_review"] = (
        merged.get("hold_for_manual_review_loan", False).fillna(False)
        | merged.get("hold_for_manual_review_profile", False).fillna(False)
    )

    log_rule(
        audit_log, "merged_dataset", "MERGE-001", "flag_data_quality",
        "applicant_id", "Consolidated credit profile generated for audit evaluation.",
        pd.Series([True] * len(merged)), merged, "applicant_id",
        "Pipeline merge step completed.",
    )

    print(f"Merged dataset shape: {merged.shape}")
    return merged


# ==========================================================================
# MAIN RUNNER
# ==========================================================================

def run_pipeline():
    audit_log = new_audit_log()

    print("=" * 80)
    print("LOADING ALL DATABASE SCHEMAS")
    print("=" * 80)

    profiles = safe_load("applicant_profiles")
    loans = safe_load("loan_applications")
    transactions = safe_load("mobile_money_transactions")
    remittances = safe_load("remittance_records")
    utilities = safe_load("utility_payments")
    coop_members = safe_load("cooperative_members")
    coop_sales = safe_load("cooperative_sales")
    documents = safe_load("document_registry")

    if profiles is not None:
        profiles = run_applicant_profile_checks(profiles, audit_log)

    if loans is not None:
        loans = run_loan_application_checks(loans, audit_log)

    if transactions is not None:
        transactions = run_mobile_money_checks(transactions, audit_log)

    if remittances is not None:
        remittances = run_remittance_checks(remittances, profiles, audit_log)

    if utilities is not None:
        utilities = run_utility_payment_checks(utilities, audit_log)

    if coop_members is not None:
        coop_members = run_cooperative_member_checks(coop_members, audit_log)

    if coop_sales is not None:
        coop_sales = run_cooperative_sales_checks(coop_sales, coop_members, audit_log)

    if documents is not None:
        documents = run_document_registry_checks(documents, audit_log)

    if profiles is not None and loans is not None:
        profiles, loans = run_cross_table_checks(
            profiles, loans, transactions, remittances, utilities,
            coop_members, coop_sales, documents, audit_log,
        )

    merged = None
    if profiles is not None and loans is not None:
        merged = build_merged_dataset(
            profiles, loans, transactions, remittances, utilities,
            coop_members, coop_sales, documents, audit_log,
        )

    audit_df = audit_log_to_dataframe(audit_log)

    print("\n" + "=" * 80)
    print("PIPELINE PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Audit log rules fired : {len(audit_df)}")

    return {
        "applicant_profiles": profiles,
        "loan_applications": loans,
        "mobile_money_transactions": transactions,
        "remittance_records": remittances,
        "utility_payments": utilities,
        "cooperative_members": coop_members,
        "cooperative_sales": coop_sales,
        "document_registry": documents,
        "merged": merged,
        "audit_trail": audit_df,
    }


if __name__ == "__main__":
    results = run_pipeline()

    if results["audit_trail"] is not None and not results["audit_trail"].empty:
        print("\nAudit trail extract for compliance reviewer:")
        print(results["audit_trail"].head(15).to_string(index=False))
    print(results)


    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True)

    for name, df in results.items():

        output_file = output_folder / f"{name}.csv"

        df.to_csv(
            output_file,
            index=False
        )

        print(f"Saved: {output_file}")
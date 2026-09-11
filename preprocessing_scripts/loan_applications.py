"""
===========================================================================
Loan Applications Validator
===========================================================================

Checks:
    - Dataset information
    - Null values
    - Duplicate rows
    - application_id format
    - applicant_id matching / 1:1 relationship
    - Application dates (AD / BS)
    - Loan purpose
    - Requested amount (mixed-type parsing)
    - Requested tenure
    - Address fields (province / district / municipality / ward)
    - Collateral
    - Wallet / remittance / cooperative flags
    - Existing loan count
    - Credit bureau score
    - NRB blacklist / AML flags
    - Document completeness score
    - Income agent outputs (estimate / confidence)
    - Credit score / score band
    - Compliance status / flags
    - Final decision / approved amount
    - Interest rate / interest tier
    - Processing time
    - Data split
    - Noise columns

COLUMN-EXISTENCE GUARD
-----------------------
Every check function verifies its required columns are present in the
dataframe before running. If a required column is missing, the check is
skipped and a message is printed instead of raising a KeyError.

NOTE ON CROSS-TABLE CHECKS
----------------------------
This script only validates loan_applications.csv in isolation. The
following checks CANNOT be done here and would need a join against other
tables (flagged inline with "CROSS-TABLE CHECK NEEDED"):

    1. applicant_id existence          -> must exist in applicant_profiles.csv
    2. Every applicant_id appearing
       exactly once across BOTH tables -> the data model states a strict
       1:1 relationship between applicant_profiles and loan_applications
    3. province_en / district_en /
       municipality_en / ward_no       -> should match applicant_profiles.csv
                                           for the same applicant (these are
                                           described as "replicated from
                                           profile")
    4. has_esewa_account / has_khalti_account /
       remittance_receiving / cooperative_member -> should match
                                           applicant_profiles.csv (replicated
                                           columns)
    5. cooperative_id                  -> should match a real row in
                                           cooperative_members.csv

"""

import pandas as pd
from pathlib import Path
import re
import json


# ==========================================================================
# Configuration
# ==========================================================================

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "loan_applications.csv"
)

MAX_PRINT_ROWS = 10

EXPECTED_LOAN_PURPOSES = {
    "agricultural_input", "small_trade", "livestock_purchase",
    "home_repair", "microenterprise_startup", "education",
    "agri_land_development", "irrigation_equipment",
    "vehicle_purchase", "emergency_medical"
}

EXPECTED_TENURES = {6, 12, 18, 24, 36}

EXPECTED_PROVINCES = {
    "Koshi", "Madhesh", "Bagmati", "Gandaki",
    "Lumbini", "Karnali", "Sudurpashchim"
}

EXPECTED_RURAL_URBAN = {"rural", "semi_urban", "urban"}

EXPECTED_COLLATERAL_TYPES = {"land", "none"}

EXPECTED_SCORE_BANDS = {"poor", "fair", "good", "very_good", "excellent"}

EXPECTED_COMPLIANCE_STATUSES = {"pass", "flag"}

EXPECTED_FINAL_DECISIONS = {"approve", "refer", "conditional", "reject"}

EXPECTED_INTEREST_TIERS = {"base", "premium", "subprime"}

EXPECTED_DATA_SPLITS = {"train", "val", "public_test"}

# score_band -> (min_score, max_score) inclusive
SCORE_BAND_RANGES = {
    "poor": (300, 449),
    "fair": (450, 579),
    "good": (580, 669),
    "very_good": (670, 739),
    "excellent": (740, 850),
}


# ==========================================================================
# Column-Existence Guard
# ==========================================================================

def require_columns(df, columns, check_name):

    missing = [c for c in columns if c not in df.columns]

    if missing:

        print(f"\nSKIPPING '{check_name}' — missing column(s): {missing}")

    return missing


# ==========================================================================
# Load Dataset
# ==========================================================================

def load_dataset():

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    print(f"Dataset Path : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded successfully.")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    return df


# ==========================================================================
# Dataset Information
# ==========================================================================

def dataset_information(df):

    print("\n")
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)

    print(df.info())

    print("\nColumns")

    for column in df.columns:
        print(f"  • {column}")


# ==========================================================================
# Null Value Check
# ==========================================================================

def check_nulls(
    df,
    drop_null_rows=False
):

    print("\n")
    print("=" * 80)
    print("NULL VALUE CHECK")
    print("=" * 80)

    summary = pd.DataFrame({
        "Null Count": df.isnull().sum()
    })

    summary["Percentage"] = (
        summary["Null Count"]
        / len(df)
        * 100
    ).round(2)

    print(summary)

    print("\nNOTE: requested_tenure_months, cooperative_id, credit_bureau_score,")
    print("doc_completeness_score, final_decision, and approved_amount_nrs are")
    print("expected to contain NULLs by design (see data dictionary).")

    if drop_null_rows:

        before = len(df)

        df = df.dropna()

        after = len(df)

        print()

        print(f"Dropped {before-after:,} rows containing NULL values.")

        print(f"Remaining rows : {after:,}")

    return df


# ==========================================================================
# Duplicate Check
# ==========================================================================

def check_duplicates(
    df,
    drop_duplicates=False
):

    print("\n")
    print("=" * 80)
    print("DUPLICATE ROW CHECK")
    print("=" * 80)

    duplicate_rows = df.duplicated()

    count = duplicate_rows.sum()

    print(f"Exact duplicate rows : {count:,}")
    print(f"Percentage           : {count / len(df) * 100:.2f}%")

    if count:

        print("\nExample duplicate rows:\n")

        print(
            df.loc[
                duplicate_rows
            ].head(MAX_PRINT_ROWS)
        )

    if "applicant_id" in df.columns:

        print("\nChecking for applicants with more than one application...")

        dup_applicant = df["applicant_id"].duplicated()

        print(f"Rows with a duplicate applicant_id : {dup_applicant.sum():,}")

        if "_noise_duplicate_app" in df.columns:

            flagged = df["_noise_duplicate_app"] == True

            mismatch = flagged != dup_applicant

            print(f"Mismatch vs _noise_duplicate_app flag : {mismatch.sum():,}")

        if dup_applicant.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    dup_applicant,
                    ["application_id", "applicant_id"]
                ].head(MAX_PRINT_ROWS)
            )

    if drop_duplicates:

        df = df.drop_duplicates()

        print()

        print("Exact duplicate rows removed.")

        print(f"Remaining rows : {len(df):,}")

    return df


# ==========================================================================
# application_id / applicant_id Validation
# ==========================================================================

def check_ids(
    df,
    drop_invalid_format=False,
    drop_duplicate_ids=False
):

    print("\n")
    print("=" * 80)
    print("APPLICATION ID / APPLICANT ID VALIDATION")
    print("=" * 80)

    if not require_columns(df, ["application_id"], "application_id format check"):

        valid_format = (
            df["application_id"]
            .astype(str)
            .str.match(r"^LA-\d{4}-\d{6}$")
        )

        invalid = ~valid_format

        print(f"Invalid application_id format : {invalid.sum():,}")

        if invalid.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    invalid,
                    ["application_id"]
                ].head(MAX_PRINT_ROWS)
            )

        if drop_invalid_format:

            df = df.loc[~invalid].copy()

            print("\nInvalid application IDs removed.")

        print("\nChecking duplicate application_id values (should be unique)...")

        duplicate_ids = df["application_id"].duplicated()

        print(f"Duplicate application IDs : {duplicate_ids.sum():,}")

        if drop_duplicate_ids:

            df = df.loc[~duplicate_ids].copy()

            print("\nDuplicate application IDs removed.")

    if not require_columns(df, ["applicant_id"], "applicant_id format check"):

        valid_format = (
            df["applicant_id"]
            .astype(str)
            .str.match(r"^AP-\d{6}$")
        )

        invalid = ~valid_format

        print(f"\nInvalid applicant_id format : {invalid.sum():,}")

        if invalid.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    invalid,
                    ["applicant_id"]
                ].head(MAX_PRINT_ROWS)
            )

        print("\nCROSS-TABLE CHECK NEEDED: confirm every applicant_id exists")
        print("in applicant_profiles.csv, and that the relationship is a")
        print("strict 1:1 (every profile has exactly one application and")
        print("vice versa).")

    return df


# ==========================================================================
# Application Date Check
# ==========================================================================

def check_application_dates(df):

    print("\n")
    print("=" * 80)
    print("APPLICATION DATE CHECK")
    print("=" * 80)

    if "application_date_ad" in df.columns:

        parsed_ad = pd.to_datetime(
            df["application_date_ad"],
            format="%Y-%m-%d",
            errors="coerce"
        )

        invalid_ad = parsed_ad.isna()

        print(f"Unable to parse application_date_ad : {invalid_ad.sum():,}")

        not_expected = (
            parsed_ad.notna()
            &
            (parsed_ad != pd.Timestamp("2024-04-15"))
        )

        print("\nNOTE: the data dictionary states all records should have")
        print("application_date_ad = '2024-04-15'.")

        print(f"Rows deviating from 2024-04-15 : {not_expected.sum():,}")

        if not_expected.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    not_expected,
                    ["application_id", "application_date_ad"]
                ].head(MAX_PRINT_ROWS)
            )

    else:

        print("SKIPPING application_date_ad check — column missing.")

    if "application_date_bs" in df.columns:

        valid_bs = (
            df["application_date_bs"]
            .astype(str)
            .str.match(r"^\d{4}-\d{2}-\d{2}$")
        )

        print(f"\nInvalid application_date_bs format : {(~valid_bs).sum():,}")

    else:

        print("\nSKIPPING application_date_bs check — column missing.")


# ==========================================================================
# Loan Purpose Check
# ==========================================================================

def check_loan_purpose(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("LOAN PURPOSE CHECK")
    print("=" * 80)

    if require_columns(df, ["loan_purpose"], "LOAN PURPOSE CHECK"):
        return df

    distribution = (
        df["loan_purpose"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Loan Purpose Distribution:\n")
    print(distribution)

    invalid = ~df["loan_purpose"].isin(EXPECTED_LOAN_PURPOSES)

    print(f"\nInvalid loan_purpose values : {invalid.sum():,}")

    if drop_invalid:

        df = df.loc[~invalid].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Requested Amount Check
# ==========================================================================

def check_requested_amount(
    df,
    drop_unparseable=False
):

    print("\n")
    print("=" * 80)
    print("REQUESTED AMOUNT CHECK")
    print("=" * 80)

    if require_columns(df, ["requested_amount_nrs"], "REQUESTED AMOUNT CHECK"):
        return df

    raw = df["requested_amount_nrs"].astype(str)

    is_string_formatted = raw.str.contains(r"Rs\.|,", regex=True)

    print(f"Rows stored as formatted string ('Rs. 200,000' style) : "
          f"{is_string_formatted.sum():,}")

    if "_noise_amount_string" in df.columns:

        flagged = df["_noise_amount_string"] == True

        mismatch = flagged != is_string_formatted

        print(f"Mismatch vs _noise_amount_string flag : {mismatch.sum():,}")

    parsed = pd.to_numeric(
        raw.str.replace("Rs. ", "", regex=False)
           .str.replace("Rs.", "", regex=False)
           .str.replace(",", "", regex=False)
           .str.strip(),
        errors="coerce"
    )

    unparseable = parsed.isna()

    print(f"\nUnable to parse (after cleaning) : {unparseable.sum():,}")

    if unparseable.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                unparseable,
                ["application_id", "requested_amount_nrs"]
            ].head(MAX_PRINT_ROWS)
        )

    negative = parsed < 0

    print(f"\nNegative amounts : {negative.sum():,}")
    print(f"Percentage        : {negative.mean()*100:.2f}%")

    if "_noise_negative_amount" in df.columns:

        flagged = df["_noise_negative_amount"] == True

        mismatch = flagged != negative

        print(f"Mismatch vs _noise_negative_amount flag : {mismatch.sum():,}")

    print("\nParsed Amount Statistics (abs value)\n")

    print(parsed.abs().describe())

    if drop_unparseable:

        df = df.loc[
            ~unparseable
        ].copy()

        print("\nUnparseable rows removed.")

    return df


# ==========================================================================
# Requested Tenure Check
# ==========================================================================

def check_requested_tenure(df):

    print("\n")
    print("=" * 80)
    print("REQUESTED TENURE CHECK")
    print("=" * 80)

    if require_columns(df, ["requested_tenure_months"], "REQUESTED TENURE CHECK"):
        return

    tenure = pd.to_numeric(df["requested_tenure_months"], errors="coerce")

    null_tenure = df["requested_tenure_months"].isna()

    print(f"NULL tenure values : {null_tenure.sum():,}")
    print(f"Percentage         : {null_tenure.mean()*100:.2f}%")

    if "_noise_tenure_missing" in df.columns:

        flagged = df["_noise_tenure_missing"] == True

        mismatch = flagged != null_tenure

        print(f"Mismatch vs _noise_tenure_missing flag : {mismatch.sum():,}")

    invalid = (
        tenure.notna()
        &
        ~tenure.isin(EXPECTED_TENURES)
    )

    print(f"\nInvalid (non-NULL, non-enum) tenure values : {invalid.sum():,}")

    if invalid.sum():

        print("\nUnexpected values:\n")

        print(
            df.loc[invalid, "requested_tenure_months"]
            .value_counts(dropna=False)
        )


# ==========================================================================
# Address Field Check
# ==========================================================================

def check_address_fields(df):

    print("\n")
    print("=" * 80)
    print("ADDRESS FIELD CHECK")
    print("=" * 80)

    print("NOTE: these fields are replicated from applicant_profiles.csv.")
    print("CROSS-TABLE CHECK NEEDED: confirm they match the profile record")
    print("for the same applicant_id.")

    if "province_en" in df.columns:

        invalid_province = ~df["province_en"].isin(EXPECTED_PROVINCES)

        print(f"\nInvalid province_en values : {invalid_province.sum():,}")

    else:

        print("\nSKIPPING province_en check — column missing.")

    for column in ["district_en", "municipality_en"]:

        if column in df.columns:

            blank = (
                df[column].isna()
                |
                (df[column].astype(str).str.strip() == "")
            )

            print(f"\n{column}")
            print(f"  Blank / NULL rows : {blank.sum():,}")

        else:

            print(f"\nSKIPPING {column} check — column missing.")

    if "ward_no" in df.columns:

        ward = pd.to_numeric(df["ward_no"], errors="coerce")

        null_ward = ward.isna()
        zero_ward = ward == 0

        print(f"\nward_no NULL : {null_ward.sum():,}")
        print(f"ward_no zero : {zero_ward.sum():,}")

    else:

        print("\nSKIPPING ward_no check — column missing.")

    if "rural_urban" in df.columns:

        invalid_ru = ~df["rural_urban"].isin(EXPECTED_RURAL_URBAN)

        print(f"\nInvalid rural_urban values : {invalid_ru.sum():,}")

    else:

        print("\nSKIPPING rural_urban check — column missing.")


# ==========================================================================
# Collateral Check
# ==========================================================================

def check_collateral(df):

    print("\n")
    print("=" * 80)
    print("COLLATERAL CHECK")
    print("=" * 80)

    if "collateral_type" in df.columns:

        distribution = (
            df["collateral_type"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("Collateral Type Distribution:\n")
        print(distribution)

        invalid = ~df["collateral_type"].isin(EXPECTED_COLLATERAL_TYPES)

        print(f"\nInvalid collateral_type values : {invalid.sum():,}")

    else:

        print("SKIPPING collateral_type check — column missing.")

    if "collateral_value_nrs" in df.columns:

        value = pd.to_numeric(df["collateral_value_nrs"], errors="coerce")

        negative = value < 0

        print(f"\nNegative collateral_value_nrs : {negative.sum():,}")

        if "collateral_type" in df.columns:

            none_with_value = (
                (df["collateral_type"] == "none")
                &
                (value > 0)
            )

            print(f"'none' collateral_type but non-zero value : "
                  f"{none_with_value.sum():,}")

            land_with_zero = (
                (df["collateral_type"] == "land")
                &
                (value == 0)
            )

            print(f"'land' collateral_type but zero value     : "
                  f"{land_with_zero.sum():,}")

    else:

        print("\nSKIPPING collateral_value_nrs check — column missing.")

    if "requested_amount_nrs" in df.columns and "collateral_type" in df.columns:

        print("\nChecking NRB rule: loans above NRs 500,000 require")
        print("collateral_type = 'land'...")

        requested = pd.to_numeric(
            df["requested_amount_nrs"]
            .astype(str)
            .str.replace("Rs. ", "", regex=False)
            .str.replace(",", "", regex=False),
            errors="coerce"
        ).abs()

        large_no_collateral = (
            (requested > 500000)
            &
            (df["collateral_type"] != "land")
        )

        print(f"Loans > 500,000 without land collateral : "
              f"{large_no_collateral.sum():,}")

        if large_no_collateral.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    large_no_collateral,
                    ["application_id", "requested_amount_nrs", "collateral_type"]
                ].head(MAX_PRINT_ROWS)
            )


# ==========================================================================
# Replicated Flag Check (wallets / remittance / cooperative)
# ==========================================================================

def check_replicated_flags(df):

    print("\n")
    print("=" * 80)
    print("REPLICATED FLAG CHECK")
    print("=" * 80)

    print("NOTE: these boolean columns are replicated from")
    print("applicant_profiles.csv. CROSS-TABLE CHECK NEEDED to confirm they")
    print("match the profile record for the same applicant_id.")

    for column in [
        "has_esewa_account",
        "has_khalti_account",
        "remittance_receiving",
        "cooperative_member"
    ]:

        if column in df.columns:

            distribution = (
                df[column]
                .value_counts(dropna=False)
                .to_frame("Count")
            )

            distribution["Percentage"] = (
                distribution["Count"] / len(df) * 100
            ).round(2)

            print(f"\n{column} Distribution:\n")
            print(distribution)

        else:

            print(f"\nSKIPPING {column} check — column missing.")

    if "cooperative_member" in df.columns and "cooperative_id" in df.columns:

        is_member = df["cooperative_member"] == True

        coop_id_present = df["cooperative_id"].notna()

        mismatch = is_member != coop_id_present

        print(f"\ncooperative_member vs cooperative_id mismatches : "
              f"{mismatch.sum():,}")

        valid_coop_id = (
            df.loc[coop_id_present, "cooperative_id"]
            .astype(str)
            .str.match(r"^COOP-[A-Za-z]{2,4}-\d{4}$")
        )

        print(f"Invalid cooperative_id format (of non-NULL) : "
              f"{(~valid_coop_id).sum():,}")


# ==========================================================================
# Existing Loan Count Check
# ==========================================================================

def check_existing_loan_count(df):

    print("\n")
    print("=" * 80)
    print("EXISTING LOAN COUNT CHECK")
    print("=" * 80)

    if require_columns(df, ["existing_loan_count"], "EXISTING LOAN COUNT CHECK"):
        return

    count = pd.to_numeric(df["existing_loan_count"], errors="coerce")

    out_of_range = (
        count.notna()
        &
        ((count < 0) | (count > 3))
    )

    print(f"Out-of-range values (expected 0–3) : {out_of_range.sum():,}")

    print("\nDistribution\n")

    print(
        df["existing_loan_count"]
        .value_counts(dropna=False)
        .sort_index()
    )


# ==========================================================================
# Credit Bureau Score Check
# ==========================================================================

def check_credit_bureau_score(df):

    print("\n")
    print("=" * 80)
    print("CREDIT BUREAU SCORE CHECK")
    print("=" * 80)

    if require_columns(df, ["credit_bureau_score"], "CREDIT BUREAU SCORE CHECK"):
        return

    score = pd.to_numeric(df["credit_bureau_score"], errors="coerce")

    null_score = df["credit_bureau_score"].isna()

    print(f"NULL credit_bureau_score : {null_score.sum():,}")
    print(f"Percentage               : {null_score.mean()*100:.2f}%")
    print("(Data dictionary expects ~72% NULL — this is by design.)")

    out_of_range = (
        score.notna()
        &
        ((score < 300) | (score > 850))
    )

    print(f"\nOut-of-range values (expected 300–850) : {out_of_range.sum():,}")

    print("\nScore Statistics (non-NULL)\n")

    print(score.describe())


# ==========================================================================
# NRB Blacklist / AML Flag Check
# ==========================================================================

def check_blacklist_and_aml(df):

    print("\n")
    print("=" * 80)
    print("NRB BLACKLIST / AML FLAG CHECK")
    print("=" * 80)

    if "nrb_blacklist_flag" in df.columns:

        distribution = (
            df["nrb_blacklist_flag"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("nrb_blacklist_flag Distribution:\n")
        print(distribution)

        if "final_decision" in df.columns:

            print("\nChecking NRB rule: blacklisted applicants must be")
            print("rejected...")

            blacklisted_not_rejected = (
                (df["nrb_blacklist_flag"] == True)
                &
                (df["final_decision"] != "reject")
            )

            print(f"Blacklisted but not rejected : "
                  f"{blacklisted_not_rejected.sum():,}")

            if blacklisted_not_rejected.sum():

                print("\nExample rows:\n")

                print(
                    df.loc[
                        blacklisted_not_rejected,
                        ["application_id", "nrb_blacklist_flag", "final_decision"]
                    ].head(MAX_PRINT_ROWS)
                )

    else:

        print("SKIPPING nrb_blacklist_flag check — column missing.")

    if "aml_flag" in df.columns:

        distribution = (
            df["aml_flag"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("\naml_flag Distribution:\n")
        print(distribution)

    else:

        print("\nSKIPPING aml_flag check — column missing.")


# ==========================================================================
# Document Completeness Score Check
# ==========================================================================

def check_doc_completeness_score(df):

    print("\n")
    print("=" * 80)
    print("DOCUMENT COMPLETENESS SCORE CHECK")
    print("=" * 80)

    if require_columns(df, ["doc_completeness_score"], "DOCUMENT COMPLETENESS SCORE CHECK"):
        return

    score = pd.to_numeric(df["doc_completeness_score"], errors="coerce")

    null_score = df["doc_completeness_score"].isna()

    print(f"NULL doc_completeness_score : {null_score.sum():,}")
    print(f"Percentage                  : {null_score.mean()*100:.2f}%")

    out_of_range = (
        score.notna()
        &
        ((score < 0.0) | (score > 1.0))
    )

    print(f"\nOut-of-range values (expected 0.0–1.0) : {out_of_range.sum():,}")

    print("\nScore Statistics (non-NULL)\n")

    print(score.describe())


# ==========================================================================
# Income Agent Output Check
# ==========================================================================

def check_income_agent_outputs(df):

    print("\n")
    print("=" * 80)
    print("INCOME AGENT OUTPUT CHECK")
    print("=" * 80)

    if "income_agent_monthly_est" in df.columns:

        income = pd.to_numeric(df["income_agent_monthly_est"], errors="coerce")

        out_of_range = (
            income.notna()
            &
            ((income < 3000) | (income > 200000))
        )

        print(f"income_agent_monthly_est out-of-range (expected 3,000–")
        print(f"200,000) : {out_of_range.sum():,}")

        print("\nIncome Estimate Statistics\n")
        print(income.describe())

    else:

        print("SKIPPING income_agent_monthly_est check — column missing.")

    if "income_confidence" in df.columns:

        confidence = pd.to_numeric(df["income_confidence"], errors="coerce")

        out_of_range = (
            confidence.notna()
            &
            ((confidence < 0.05) | (confidence > 0.97))
        )

        print(f"\nincome_confidence out-of-range (expected 0.05–0.97) : "
              f"{out_of_range.sum():,}")

        print("\nConfidence Statistics\n")
        print(confidence.describe())

        if "approved_amount_nrs" in df.columns and "requested_amount_nrs" in df.columns:

            print("\nChecking haircut rule: income_confidence < 0.70 should")
            print("imply approved_amount_nrs ≈ requested_amount_nrs × 0.85 ")
            print("(for approved/conditional loans)...")

            requested = pd.to_numeric(
                df["requested_amount_nrs"]
                .astype(str)
                .str.replace("Rs. ", "", regex=False)
                .str.replace(",", "", regex=False),
                errors="coerce"
            ).abs()

            approved = pd.to_numeric(df["approved_amount_nrs"], errors="coerce")

            expected_haircut = requested * 0.85

            low_confidence = confidence < 0.70

            both_present = low_confidence & approved.notna() & requested.notna()

            diff_pct = (
                (approved - expected_haircut).abs()
                /
                expected_haircut.replace(0, pd.NA)
            ) * 100

            haircut_mismatch = both_present & (diff_pct > 2.0)

            print(f"Low-confidence rows with approved amount not matching")
            print(f"the haircut formula (>2% diff) : {haircut_mismatch.sum():,}")

    else:

        print("\nSKIPPING income_confidence check — column missing.")


# ==========================================================================
# Credit Score / Score Band Check
# ==========================================================================

def check_credit_score_and_band(df):

    print("\n")
    print("=" * 80)
    print("CREDIT SCORE / SCORE BAND CHECK")
    print("=" * 80)

    if "credit_score" in df.columns:

        score = pd.to_numeric(df["credit_score"], errors="coerce")

        out_of_range = (
            score.notna()
            &
            ((score < 300) | (score > 850))
        )

        print(f"credit_score out-of-range (expected 300–850) : "
              f"{out_of_range.sum():,}")

        print("\nCredit Score Statistics\n")
        print(score.describe())

    else:

        print("SKIPPING credit_score check — column missing.")

    if "score_band" in df.columns:

        distribution = (
            df["score_band"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("\nScore Band Distribution:\n")
        print(distribution)

        invalid = ~df["score_band"].isin(EXPECTED_SCORE_BANDS)

        print(f"\nInvalid score_band values : {invalid.sum():,}")

        if "credit_score" in df.columns:

            print("\nChecking score_band matches credit_score range...")

            def band_matches(row):

                band = row["score_band"]
                score_val = row["credit_score"]

                if band not in SCORE_BAND_RANGES or pd.isna(score_val):
                    return True  # can't evaluate; don't flag

                low, high = SCORE_BAND_RANGES[band]

                return low <= score_val <= high

            matches = df.apply(band_matches, axis=1)

            mismatch = ~matches

            print(f"Rows where score_band doesn't match credit_score : "
                  f"{mismatch.sum():,}")

            if mismatch.sum():

                print("\nExample rows:\n")

                print(
                    df.loc[
                        mismatch,
                        ["application_id", "credit_score", "score_band"]
                    ].head(MAX_PRINT_ROWS)
                )

    else:

        print("\nSKIPPING score_band check — column missing.")


# ==========================================================================
# Compliance Status / Flags Check
# ==========================================================================

def check_compliance(df):

    print("\n")
    print("=" * 80)
    print("COMPLIANCE STATUS / FLAGS CHECK")
    print("=" * 80)

    if "compliance_status" in df.columns:

        distribution = (
            df["compliance_status"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("Compliance Status Distribution:\n")
        print(distribution)

        invalid = ~df["compliance_status"].isin(EXPECTED_COMPLIANCE_STATUSES)

        print(f"\nInvalid compliance_status values (should never see 'veto') "
              f": {invalid.sum():,}")

    else:

        print("SKIPPING compliance_status check — column missing.")

    if "compliance_flags" in df.columns:

        def is_valid_json_array(value):

            if pd.isna(value):
                return False

            try:
                parsed = json.loads(value)
                return isinstance(parsed, list)
            except (json.JSONDecodeError, TypeError):
                return False

        valid_json = df["compliance_flags"].apply(is_valid_json_array)

        print(f"\nRows where compliance_flags isn't a valid JSON array : "
              f"{(~valid_json).sum():,}")

        if (~valid_json).sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    ~valid_json,
                    ["application_id", "compliance_flags"]
                ].head(MAX_PRINT_ROWS)
            )

        if "compliance_status" in df.columns:

            print("\nChecking compliance_status <-> compliance_flags "
                  "consistency...")

            empty_flags = (
                df["compliance_flags"].astype(str).isin(["[]", "[ ]"])
            )

            pass_with_flags = (
                (df["compliance_status"] == "pass")
                &
                (~empty_flags)
            )

            flag_with_empty = (
                (df["compliance_status"] == "flag")
                &
                empty_flags
            )

            print(f"'pass' status but non-empty flags array : "
                  f"{pass_with_flags.sum():,}")
            print(f"'flag' status but empty flags array     : "
                  f"{flag_with_empty.sum():,}")

    else:

        print("\nSKIPPING compliance_flags check — column missing.")


# ==========================================================================
# Final Decision / Approved Amount Check
# ==========================================================================

def check_final_decision(df):

    print("\n")
    print("=" * 80)
    print("FINAL DECISION / APPROVED AMOUNT CHECK")
    print("=" * 80)

    if "final_decision" in df.columns:

        distribution = (
            df["final_decision"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("Final Decision Distribution:\n")
        print(distribution)

        null_decision = df["final_decision"].isna()

        invalid = (
            df["final_decision"].notna()
            &
            ~df["final_decision"].isin(EXPECTED_FINAL_DECISIONS)
        )

        print(f"\nNULL final_decision       : {null_decision.sum():,}")
        print(f"Invalid (non-NULL) values : {invalid.sum():,}")

        if "_noise_decision_missing" in df.columns:

            flagged = df["_noise_decision_missing"] == True

            mismatch = flagged != null_decision

            print(f"Mismatch vs _noise_decision_missing flag : "
                  f"{mismatch.sum():,}")

    else:

        print("SKIPPING final_decision check — column missing.")

    if "approved_amount_nrs" in df.columns and "final_decision" in df.columns:

        print("\nChecking approved_amount_nrs is NULL iff decision is")
        print("'refer' or 'reject'...")

        should_be_null = df["final_decision"].isin(["refer", "reject"])

        is_null = df["approved_amount_nrs"].isna()

        mismatch = should_be_null != is_null

        print(f"Mismatches : {mismatch.sum():,}")

        if mismatch.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    mismatch,
                    ["application_id", "final_decision", "approved_amount_nrs"]
                ].head(MAX_PRINT_ROWS)
            )

        if "requested_amount_nrs" in df.columns:

            print("\nChecking approved_amount_nrs does not exceed")
            print("requested_amount_nrs...")

            requested = pd.to_numeric(
                df["requested_amount_nrs"]
                .astype(str)
                .str.replace("Rs. ", "", regex=False)
                .str.replace(",", "", regex=False),
                errors="coerce"
            ).abs()

            approved = pd.to_numeric(df["approved_amount_nrs"], errors="coerce")

            exceeds = approved > requested

            print(f"Rows where approved > requested : {exceeds.sum():,}")

    elif "approved_amount_nrs" in df.columns:

        print("\nSKIPPING approved vs decision consistency check — "
              "final_decision column missing.")


# ==========================================================================
# Interest Rate / Interest Tier Check
# ==========================================================================

def check_interest_rate_and_tier(df):

    print("\n")
    print("=" * 80)
    print("INTEREST RATE / INTEREST TIER CHECK")
    print("=" * 80)

    if "interest_rate_pct" in df.columns:

        rate = pd.to_numeric(df["interest_rate_pct"], errors="coerce")

        unparseable = rate.isna()

        print(f"Unable to parse interest_rate_pct : {unparseable.sum():,}")

        if "final_decision" in df.columns:

            approved_like = df["final_decision"].isin(["approve", "conditional"])

            print("\nChecking corridor (10.0–15.0) for approved/conditional")
            print("loans, and 0.0 for rejected/referred/NULL-decision loans...")

            out_of_corridor = (
                approved_like
                &
                rate.notna()
                &
                ((rate < 10.0) | (rate > 15.0))
            )

            print(f"Approved/conditional loans outside 10–15% corridor : "
                  f"{out_of_corridor.sum():,}")

            if "_noise_interest_oor" in df.columns:

                flagged = df["_noise_interest_oor"] == True

                print(f"Rows flagged as _noise_interest_oor : "
                      f"{flagged.sum():,}")

            not_approved = ~approved_like

            non_zero_when_not_approved = (
                not_approved
                &
                (rate != 0.0)
                &
                rate.notna()
            )

            print(f"\nRejected/referred/unknown loans with non-zero rate : "
                  f"{non_zero_when_not_approved.sum():,}")

        else:

            print("\nSKIPPING corridor check — final_decision column missing.")

    else:

        print("SKIPPING interest_rate_pct check — column missing.")

    if "interest_tier" in df.columns:

        distribution = (
            df["interest_tier"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("\nInterest Tier Distribution:\n")
        print(distribution)

        invalid = ~df["interest_tier"].isin(EXPECTED_INTEREST_TIERS)

        print(f"\nInvalid interest_tier values : {invalid.sum():,}")

        if "credit_score" in df.columns:

            print("\nChecking interest_tier matches credit_score thresholds")
            print("(base >= 670, premium 580–669, subprime < 580)...")

            score = pd.to_numeric(df["credit_score"], errors="coerce")

            expected_tier = pd.cut(
                score,
                bins=[-float("inf"), 579, 669, float("inf")],
                labels=["subprime", "premium", "base"]
            )

            tier_mismatch = (
                score.notna()
                &
                (df["interest_tier"].astype(str) != expected_tier.astype(str))
            )

            print(f"Rows where interest_tier doesn't match credit_score : "
                  f"{tier_mismatch.sum():,}")

    else:

        print("\nSKIPPING interest_tier check — column missing.")


# ==========================================================================
# Processing Time Check
# ==========================================================================

def check_processing_time(df):

    print("\n")
    print("=" * 80)
    print("PROCESSING TIME CHECK")
    print("=" * 80)

    if require_columns(df, ["processing_time_seconds"], "PROCESSING TIME CHECK"):
        return

    seconds = pd.to_numeric(df["processing_time_seconds"], errors="coerce")

    out_of_range = (
        seconds.notna()
        &
        ((seconds < 12) | (seconds > 200))
    )

    print(f"Out-of-range values (expected 12–200) : {out_of_range.sum():,}")

    print("\nProcessing Time Statistics\n")

    print(seconds.describe())


# ==========================================================================
# Data Split Check
# ==========================================================================

def check_data_split(df):

    print("\n")
    print("=" * 80)
    print("DATA SPLIT CHECK")
    print("=" * 80)

    if require_columns(df, ["data_split"], "DATA SPLIT CHECK"):
        return

    distribution = (
        df["data_split"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Data Split Distribution:\n")
    print(distribution)

    invalid = ~df["data_split"].isin(EXPECTED_DATA_SPLITS)

    print(f"\nInvalid data_split values : {invalid.sum():,}")


# ==========================================================================
# Noise Columns Check
# ==========================================================================

def check_noise_columns(df):

    print("\n")
    print("=" * 80)
    print("NOISE COLUMN SUMMARY")
    print("=" * 80)

    columns = [

        "_noise_amount_string",
        "_noise_negative_amount",
        "_noise_tenure_missing",
        "_noise_decision_missing",
        "_noise_interest_oor",
        "_noise_duplicate_app"

    ]

    for column in columns:

        if column not in df.columns:

            print(f"\nSKIPPING {column} — column missing.")
            continue

        print(f"\n{column}")

        distribution = (
            df[column]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"]
            / len(df)
            * 100
        ).round(2)

        print(distribution)


# ==========================================================================
# Run All Checks
# ==========================================================================

def run_all_checks():

    df = load_dataset()

    dataset_information(df)

    df = check_nulls(
        df,
        drop_null_rows=False
    )

    df = check_duplicates(
        df,
        drop_duplicates=False
    )

    df = check_ids(
        df,
        drop_invalid_format=True,
        drop_duplicate_ids=False
    )

    check_application_dates(df)

    df = check_loan_purpose(
        df,
        drop_invalid=False
    )

    df = check_requested_amount(
        df,
        drop_unparseable=False
    )

    check_requested_tenure(df)

    check_address_fields(df)

    check_collateral(df)

    check_replicated_flags(df)

    check_existing_loan_count(df)

    check_credit_bureau_score(df)

    check_blacklist_and_aml(df)

    check_doc_completeness_score(df)

    check_income_agent_outputs(df)

    check_credit_score_and_band(df)

    check_compliance(df)

    check_final_decision(df)

    check_interest_rate_and_tier(df)

    check_processing_time(df)

    check_data_split(df)

    check_noise_columns(df)

    print("\n")
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(f"Final Rows    : {len(df):,}")
    print(f"Final Columns : {len(df.columns)}")

    return df


# ==========================================================================
# Main
# ==========================================================================


if __name__ == "__main__":

    df = run_all_checks()

    # ------------------------------------------------------
    # Save cleaned dataset
    # ------------------------------------------------------

    output_folder = (
        Path(__file__).resolve().parent.parent
        / "cleaned_datasets"
    )

    output_folder.mkdir(exist_ok=True)

    pattern = re.compile(
        r"loan_applications_ver(\d+)\.csv"
    )

    versions = []

    for file in output_folder.glob("loan_applications_ver*.csv"):

        match = pattern.fullmatch(file.name)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1

    output_file = (
        output_folder
        / f"loan_applications_ver{next_version}.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("DATASET SAVED")
    print("=" * 80)
    print(f"Version : {next_version}")
    print(f"File    : {output_file}")
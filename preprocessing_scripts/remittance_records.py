"""
===========================================================================
Remittance Records Validator
===========================================================================

Checks:
    - Dataset information
    - Null values
    - Duplicate rows
    - remittance_id format
    - applicant_id matching
    - Transfer service values
    - Transfer / received dates
    - Foreign currency amount
    - Foreign currency code
    - Amount NRs
    - Exchange rate
    - Transaction fee
    - Disbursement mode
    - Relationship to receiver
    - Purpose declared
    - Name match score
    - Geolocation (receiver district)
    - Noise columns

NOTE ON CROSS-PARQUET CHECKS
-----------------------------
This script deliberately only validates remittance_records.parquet in
isolation. The following checks CANNOT be done here and would need a join
against other datasets (flagged inline with "CROSS-PARQUET CHECK NEEDED"):

    1. applicant_id existence       -> must exist in applicant_profiles.parquet
    2. receiver_name_en             -> should match (or fuzzy-match) the name
                                        in applicant_profiles.parquet
    3. receiver_district_en /
       receiver_municipality_en     -> should match applicant_profiles.parquet
                                        address fields
    4. sender_name / relationship   -> should match a known family member
                                        record (if a family_members /
                                        household table exists) to validate
                                        name_match_score is computed
                                        consistently
    5. foreign_currency_code        -> should be cross-checked against the
                                        expected currency for
                                        sender_country_code (e.g. Qatar ->
                                        QAR, Saudi Arabia -> SAR) using a
                                        country->currency reference table,
                                        rather than just checking it's a
                                        plausible ISO 4217 code in isolation

"""

import pandas as pd
from pathlib import Path
import re


# ==========================================================================
# Configuration
# ==========================================================================

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "remittance_records.parquet"
)

MAX_PRINT_ROWS = 10

EXPECTED_TRANSFER_SERVICES = {
    "IME Money",
    "Prabhu Money",
    "Himal Remit",
    "Western Union NP"
}

EXPECTED_DISBURSEMENT_MODES = {
    "bank_deposit",
    "mobile_wallet",
    "cash_pickup"
}

EXPECTED_COUNTRY_CODES = {
    "QA",  # Qatar
    "SA",  # Saudi Arabia
    "AE",  # UAE
    "MY",  # Malaysia
    "IN",  # India
    "KR",  # South Korea
    "US",  # USA
    "JP"   # Japan
}


# ==========================================================================
# Load Dataset
# ==========================================================================

def load_dataset():

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    print(f"Dataset Path : {DATASET_PATH}")

    df = pd.read_parquet(DATASET_PATH)

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

    percentage = count / len(df) * 100

    print(f"Duplicate Rows : {count:,}")
    print(f"Percentage     : {percentage:.2f}%")

    if count:

        print("\nExample duplicate rows:\n")

        print(
            df.loc[
                duplicate_rows
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Cross-check against the explicit _noise_duplicate flag
    # ------------------------------------------------------

    if "_noise_duplicate" in df.columns:

        print("\nComparing against _noise_duplicate flag...")

        flagged = df["_noise_duplicate"] == True

        print(f"Rows flagged as duplicate : {flagged.sum():,}")

        mismatch = flagged != duplicate_rows

        print(f"Mismatch vs pandas .duplicated() : {mismatch.sum():,}")

        if mismatch.sum():

            print("\nExample mismatched rows:\n")

            print(
                df.loc[
                    mismatch,
                    [
                        "remittance_id",
                        "_noise_duplicate"
                    ]
                ].head(MAX_PRINT_ROWS)
            )

    if drop_duplicates:

        df = df.drop_duplicates()

        print()

        print("Duplicate rows removed.")

        print(f"Remaining rows : {len(df):,}")

    return df


# ==========================================================================
# remittance_id Validation
# ==========================================================================

def check_remittance_id(
    df,
    drop_invalid_format=False,
    drop_mismatched_applicant=False,
    drop_duplicate_ids=False
):

    print("\n")
    print("=" * 80)
    print("REMITTANCE ID VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------
    # Format Check
    # ------------------------------------------------------

    print("\nChecking remittance_id format...")

    valid_format = (
        df["remittance_id"]
        .astype(str)
        .str.match(r"^REM-AP-\d{6}-\d{4}-\d{2}$")
    )

    invalid = ~valid_format

    print(f"Invalid remittance IDs : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                ["remittance_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid_format:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid remittance IDs removed.")

    # ------------------------------------------------------
    # applicant_id match (format-level only)
    # ------------------------------------------------------

    print("\nChecking applicant_id consistency with remittance_id...")
    print("NOTE: this only checks internal consistency between the two")
    print("columns in this file. Whether applicant_id actually exists is a")
    print("CROSS-PARQUET CHECK NEEDED against applicant_profiles.csv.")

    extracted = (
        df["remittance_id"]
        .str.extract(r"(AP-\d{6})")[0]
    )

    mismatch = (
        extracted
        !=
        df["applicant_id"]
    )

    print(f"\nMismatched IDs : {mismatch.sum():,}")

    if mismatch.sum():

        print()

        print(
            df.loc[
                mismatch,
                [
                    "remittance_id",
                    "applicant_id"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_mismatched_applicant:

        df = df.loc[
            ~mismatch
        ].copy()

        print("\nMismatched rows removed.")

    # ------------------------------------------------------
    # Duplicate remittance IDs
    # ------------------------------------------------------

    print("\nChecking duplicate remittance_id values...")

    duplicate_ids = (
        df["remittance_id"]
        .duplicated()
    )

    print(f"Duplicate IDs : {duplicate_ids.sum():,}")

    if duplicate_ids.sum():

        print()

        print(
            df.loc[
                duplicate_ids,
                ["remittance_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_duplicate_ids:

        df = df.loc[
            ~duplicate_ids
        ].copy()

        print("\nDuplicate remittance IDs removed.")

    return df


# ==========================================================================
# Receiver Identity Check (in-file consistency only)
# ==========================================================================

def check_receiver_identity(df):

    print("\n")
    print("=" * 80)
    print("RECEIVER IDENTITY CHECK")
    print("=" * 80)

    print("Checking receiver_name_en / receiver_district_en / "
          "receiver_municipality_en for NULL / blank values only.")

    print("CROSS-PARQUET CHECK NEEDED: compare these fields against ")
    print("applicant_profiles.csv to confirm receiver_name_en and ")
    print("address fields match the applicant's registered profile.")

    for column in [
        "receiver_name_en",
        "receiver_district_en",
        "receiver_municipality_en"
    ]:

        blank = (
            df[column].isna()
            |
            (df[column].astype(str).str.strip() == "")
        )

        print(f"\n{column}")
        print(f"  Blank / NULL rows : {blank.sum():,}")
        print(f"  Percentage        : {blank.mean()*100:.2f}%")


# ==========================================================================
# Sender Name / Name Corruption Check
# ==========================================================================

def check_sender_name(df):

    print("\n")
    print("=" * 80)
    print("SENDER NAME CHECK")
    print("=" * 80)

    print("CROSS-PARQUET CHECK NEEDED: sender_name should ideally be ")
    print("compared against a known family-member / household reference ")
    print("table (if one exists) to properly validate name_match_score.")

    if "_noise_name_corrupt" in df.columns:

        corrupt = df["_noise_name_corrupt"] == True

        print(f"\nRows flagged as name-corrupted : {corrupt.sum():,}")
        print(f"Percentage                     : {corrupt.mean()*100:.2f}%")

        if corrupt.sum():

            print("\nExample corrupted rows:\n")

            print(
                df.loc[
                    corrupt,
                    [
                        "remittance_id",
                        "sender_name",
                        "name_match_score"
                    ]
                ].head(MAX_PRINT_ROWS)
            )

        # Sanity check: corrupted names should tend to score low
        avg_corrupt_score = df.loc[corrupt, "name_match_score"].mean()
        avg_clean_score = df.loc[~corrupt, "name_match_score"].mean()

        print(f"\nAvg name_match_score (corrupted) : {avg_corrupt_score:.3f}")
        print(f"Avg name_match_score (clean)     : {avg_clean_score:.3f}")


# ==========================================================================
# Sender Country Check
# ==========================================================================

def check_sender_country(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("SENDER COUNTRY CHECK")
    print("=" * 80)

    distribution = (
        df["sender_country_code"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Sender Country Code Distribution:\n")
    print(distribution)

    invalid = ~df["sender_country_code"].isin(EXPECTED_COUNTRY_CODES)

    print(f"\nInvalid country codes : {invalid.sum():,}")
    print(f"Percentage            : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "remittance_id",
                    "sender_country_code",
                    "sender_country_name"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Internal consistency: country_code vs country_name
    # ------------------------------------------------------

    print("\nChecking country_code <-> country_name consistency...")

    code_to_name = (
        df.dropna(subset=["sender_country_code", "sender_country_name"])
        .groupby("sender_country_code")["sender_country_name"]
        .nunique()
    )

    inconsistent_codes = code_to_name[code_to_name > 1]

    if len(inconsistent_codes):

        print("\nCountry codes mapped to multiple country names:\n")
        print(inconsistent_codes)

    else:

        print("All country codes map to a single country name.")

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Transfer Service Check
# ==========================================================================

def check_transfer_service(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("TRANSFER SERVICE CHECK")
    print("=" * 80)

    distribution = (
        df["transfer_service"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Transfer Service Distribution:\n")
    print(distribution)

    invalid = ~df["transfer_service"].isin(EXPECTED_TRANSFER_SERVICES)

    print(f"\nInvalid values : {invalid.sum():,}")
    print(f"Percentage     : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "remittance_id",
                    "transfer_service"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Date Checks (transfer_date_ad / received_date_ad)
# ==========================================================================

def check_dates(
    df,
    latest_valid=None,
    drop_invalid=False,
    drop_out_of_order=False
):

    if latest_valid is None:
        latest_valid = pd.Timestamp.now()

    print("\n")
    print("=" * 80)
    print("TRANSFER / RECEIVED DATE CHECK")
    print("=" * 80)

    latest_valid = pd.Timestamp(latest_valid)

    transfer_dates = pd.to_datetime(
        df["transfer_date_ad"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    received_dates = pd.to_datetime(
        df["received_date_ad"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    invalid_transfer = (
        transfer_dates.isna()
        |
        (transfer_dates > latest_valid)
    )

    invalid_received = (
        received_dates.isna()
        |
        (received_dates > latest_valid)
    )

    print("Transfer Date")
    print(f"  Impossible timestamps : {invalid_transfer.sum():,}")
    print(f"  Percentage            : {invalid_transfer.mean()*100:.2f}%")
    print(f"  Earliest              : {transfer_dates.min()}")
    print(f"  Latest                : {transfer_dates.max()}")

    print("\nReceived Date")
    print(f"  Impossible timestamps : {invalid_received.sum():,}")
    print(f"  Percentage            : {invalid_received.mean()*100:.2f}%")
    print(f"  Earliest              : {received_dates.min()}")
    print(f"  Latest                : {received_dates.max()}")

    invalid = invalid_transfer | invalid_received

    if invalid.sum():

        print("\nExample invalid rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "remittance_id",
                    "transfer_date_ad",
                    "received_date_ad"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Ordering check: received should be on/after transfer
    # ------------------------------------------------------

    print("\nChecking received_date_ad >= transfer_date_ad...")

    out_of_order = (
        transfer_dates.notna()
        & received_dates.notna()
        & (received_dates < transfer_dates)
    )

    print(f"Out-of-order rows : {out_of_order.sum():,}")
    print(f"Percentage        : {out_of_order.mean()*100:.2f}%")

    if out_of_order.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                out_of_order,
                [
                    "remittance_id",
                    "transfer_date_ad",
                    "received_date_ad"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nRows with impossible timestamps removed.")

    if drop_out_of_order:

        df = df.loc[
            ~out_of_order
        ].copy()

        print("\nOut-of-order rows removed.")

    return df


# ==========================================================================
# Foreign Currency Amount Check
# ==========================================================================

def check_amount_foreign_currency(
    df,
    drop_out_of_range=False
):

    print("\n")
    print("=" * 80)
    print("AMOUNT (FOREIGN CURRENCY) CHECK")
    print("=" * 80)

    amount = pd.to_numeric(
        df["amount_foreign_currency"],
        errors="coerce"
    )

    invalid = amount.isna()

    print(f"Unable to parse : {invalid.sum():,}")

    out_of_range = (
        amount.notna()
        &
        ((amount < 100.0) | (amount > 900.0))
    )

    print(f"\nOut-of-range values (expected 100.00–900.00)")
    print(f"Rows       : {out_of_range.sum():,}")
    print(f"Percentage : {out_of_range.mean()*100:.2f}%")

    if out_of_range.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                out_of_range,
                [
                    "remittance_id",
                    "amount_foreign_currency",
                    "foreign_currency_code"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nAmount Statistics\n")

    print(amount.describe())

    if drop_out_of_range:

        df = df.loc[
            ~out_of_range
        ].copy()

        print("\nOut-of-range rows removed.")

    return df


# ==========================================================================
# Foreign Currency Code Check
# ==========================================================================

def check_foreign_currency_code(df):

    print("\n")
    print("=" * 80)
    print("FOREIGN CURRENCY CODE CHECK")
    print("=" * 80)

    distribution = (
        df["foreign_currency_code"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Currency Code Distribution:\n")
    print(distribution)

    if "_noise_wrong_currency" in df.columns:

        wrong = df["_noise_wrong_currency"] == True

        print(f"\nRows flagged as wrong currency : {wrong.sum():,}")
        print(f"Percentage                     : {wrong.mean()*100:.2f}%")

    print("\nNOTE: validating that foreign_currency_code is the CORRECT")
    print("currency for a given sender_country_code requires a country ->")
    print("currency reference mapping. That mapping is not part of this")
    print("file, so only the wrong-currency noise flag is inspected here.")


# ==========================================================================
# Amount NRs Check
# ==========================================================================

def check_amount_nrs(
    df,
    drop_non_positive=False
):

    print("\n")
    print("=" * 80)
    print("AMOUNT NRS CHECK")
    print("=" * 80)

    amount = pd.to_numeric(
        df["amount_nrs"],
        errors="coerce"
    )

    invalid = amount.isna()

    print(f"Unable to parse : {invalid.sum():,}")

    non_positive = amount <= 0

    print(f"\nZero or negative amounts")
    print(f"Rows       : {non_positive.sum():,}")
    print(f"Percentage : {non_positive.mean()*100:.2f}%")

    if non_positive.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                non_positive,
                [
                    "remittance_id",
                    "amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nAmount Statistics\n")

    print(amount.describe())

    # ------------------------------------------------------
    # Consistency: amount_nrs ≈ amount_foreign_currency * exchange_rate
    # ------------------------------------------------------

    print("\nChecking amount_nrs ≈ amount_foreign_currency × exchange_rate...")

    expected_nrs = (
        pd.to_numeric(df["amount_foreign_currency"], errors="coerce")
        *
        pd.to_numeric(df["exchange_rate"], errors="coerce")
    )

    diff_pct = (
        (amount - expected_nrs).abs()
        /
        expected_nrs.replace(0, pd.NA)
    ) * 100

    mismatched = diff_pct > 1.0  # allow small rounding tolerance

    print(f"Rows with >1% discrepancy : {mismatched.sum():,}")
    print(f"Percentage                : {mismatched.mean()*100:.2f}%")

    if mismatched.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                mismatched,
                [
                    "remittance_id",
                    "amount_foreign_currency",
                    "exchange_rate",
                    "amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_non_positive:

        df = df.loc[
            ~non_positive
        ].copy()

        print("\nNon-positive rows removed.")

    return df


# ==========================================================================
# Exchange Rate Check
# ==========================================================================

def check_exchange_rate(
    df,
    drop_impossible=False
):

    print("\n")
    print("=" * 80)
    print("EXCHANGE RATE CHECK")
    print("=" * 80)

    rate = pd.to_numeric(
        df["exchange_rate"],
        errors="coerce"
    )

    invalid = rate.isna()

    print(f"Unable to parse    : {invalid.sum():,}")

    non_positive = rate <= 0

    print(f"Zero or negative   : {non_positive.sum():,}")

    print("\nExchange Rate Statistics (overall)\n")

    print(rate.describe())

    # ------------------------------------------------------
    # Per-currency stats to spot the 10x outliers
    # ------------------------------------------------------

    print("\nExchange Rate Statistics by currency:\n")

    print(
        df.assign(_rate=rate)
        .groupby("foreign_currency_code")["_rate"]
        .describe()
    )

    if "_noise_impossible_rate" in df.columns:

        flagged = df["_noise_impossible_rate"] == True

        print(f"\nRows flagged as impossible rate : {flagged.sum():,}")
        print(f"Percentage                       : {flagged.mean()*100:.2f}%")

        if flagged.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    flagged,
                    [
                        "remittance_id",
                        "foreign_currency_code",
                        "exchange_rate"
                    ]
                ].head(MAX_PRINT_ROWS)
            )

    if drop_impossible and "_noise_impossible_rate" in df.columns:

        df = df.loc[
            ~(df["_noise_impossible_rate"] == True)
        ].copy()

        print("\nRows with impossible exchange rates removed.")

    return df


# ==========================================================================
# Transaction Fee Check
# ==========================================================================

def check_transaction_fee(df):

    print("\n")
    print("=" * 80)
    print("TRANSACTION FEE CHECK")
    print("=" * 80)

    fee = pd.to_numeric(
        df["transaction_fee_nrs"],
        errors="coerce"
    )

    not_fixed = fee != 50.0

    print(f"Rows where fee != 50.0 : {not_fixed.sum():,}")
    print(f"Percentage             : {not_fixed.mean()*100:.2f}%")

    if not_fixed.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                not_fixed,
                [
                    "remittance_id",
                    "transaction_fee_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )


# ==========================================================================
# Disbursement Mode Check
# ==========================================================================

def check_disbursement_mode(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("DISBURSEMENT MODE CHECK")
    print("=" * 80)

    distribution = (
        df["disbursement_mode"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Disbursement Mode Distribution:\n")
    print(distribution)

    invalid = ~df["disbursement_mode"].isin(EXPECTED_DISBURSEMENT_MODES)

    print(f"\nInvalid values : {invalid.sum():,}")
    print(f"Percentage     : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "remittance_id",
                    "disbursement_mode"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Relationship to Receiver Check
# ==========================================================================

def check_relationship_to_receiver(df):

    print("\n")
    print("=" * 80)
    print("RELATIONSHIP TO RECEIVER CHECK")
    print("=" * 80)

    distribution = (
        df["relationship_to_receiver"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Relationship Distribution:\n")
    print(distribution)

    print("\nNo fixed enum given in the data dictionary for this column,")
    print("so only the observed distribution is reported for review.")


# ==========================================================================
# Purpose Declared Check
# ==========================================================================

def check_purpose_declared(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("PURPOSE DECLARED CHECK")
    print("=" * 80)

    invalid = df["purpose_declared"] != "family_support"

    print(f"Rows != 'family_support' : {invalid.sum():,}")
    print(f"Percentage                : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nUnexpected values:\n")

        print(
            df.loc[
                invalid,
                "purpose_declared"
            ].value_counts(dropna=False)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Name Match Score Check
# ==========================================================================

def check_name_match_score(df):

    print("\n")
    print("=" * 80)
    print("NAME MATCH SCORE CHECK")
    print("=" * 80)

    score = pd.to_numeric(
        df["name_match_score"],
        errors="coerce"
    )

    out_of_range = (
        score.notna()
        &
        ((score < 0.0) | (score > 1.0))
    )

    print(f"Out-of-range values (expected 0.0–1.0)")
    print(f"Rows       : {out_of_range.sum():,}")
    print(f"Percentage : {out_of_range.mean()*100:.2f}%")

    if out_of_range.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                out_of_range,
                [
                    "remittance_id",
                    "name_match_score"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nScore Statistics\n")

    print(score.describe())

    below_threshold = score < 0.80

    print(f"\nRows below soft-flag threshold (< 0.80) : {below_threshold.sum():,}")
    print(f"Percentage                               : {below_threshold.mean()*100:.2f}%")

    print("\nCROSS-PARQUET CHECK NEEDED: to fully validate this score, ")
    print("sender_name should be re-matched against a household / family ")
    print("member reference table, if one exists, rather than trusting the ")
    print("score as stored.")


# ==========================================================================
# Geolocation Check (receiver district)
# ==========================================================================

def check_geolocation(df):

    print("\n")
    print("=" * 80)
    print("GEOLOCATION CHECK (RECEIVER DISTRICT)")
    print("=" * 80)

    distribution = (
        df["receiver_district_en"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print(distribution)

    print("\nCROSS-PARQUET CHECK NEEDED: confirm receiver_district_en / ")
    print("receiver_municipality_en match the address on file in ")
    print("applicant_profiles.parquet for the same applicant_id.")


# ==========================================================================
# Noise Columns Check
# ==========================================================================

def check_noise_columns(df):

    print("\n")
    print("=" * 80)
    print("NOISE COLUMN SUMMARY")
    print("=" * 80)

    columns = [

        "_noise_name_corrupt",
        "_noise_wrong_currency",
        "_noise_impossible_rate",
        "_noise_duplicate"

    ]

    for column in columns:

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
        drop_duplicates=True
    )

    df = check_remittance_id(
        df,
        drop_invalid_format=True,
        drop_mismatched_applicant=False,
        drop_duplicate_ids=True
    )

    check_receiver_identity(df)

    check_sender_name(df)

    df = check_sender_country(
        df,
        drop_invalid=False
    )

    df = check_transfer_service(
        df,
        drop_invalid=False
    )

    df = check_dates(
        df,
        latest_valid="2024-12-31",
        drop_invalid=True,
        drop_out_of_order=False
    )

    df = check_amount_foreign_currency(
        df,
        drop_out_of_range=True
    )

    check_foreign_currency_code(df)

    df = check_amount_nrs(
        df,
        drop_non_positive=False
    )

    df = check_exchange_rate(
        df,
        drop_impossible=False
    )

    check_transaction_fee(df)

    df = check_disbursement_mode(
        df,
        drop_invalid=False
    )

    check_relationship_to_receiver(df)

    df = check_purpose_declared(
        df,
        drop_invalid=False
    )

    check_name_match_score(df)

    check_geolocation(df)

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
        r"remittance_records_ver(\d+)\.parquet"
    )

    versions = []

    for file in output_folder.glob("remittance_records_ver*.parquet"):

        match = pattern.fullmatch(file.name)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1

    output_file = (
        output_folder
        / f"remittance_records_ver{next_version}.parquet"
    )

    df.to_parquet(
        output_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("DATASET SAVED")
    print("=" * 80)
    print(f"Version : {next_version}")
    print(f"File    : {output_file}")
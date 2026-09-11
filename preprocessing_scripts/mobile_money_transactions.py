"""
===========================================================================
Mobile Money Transactions Validator
===========================================================================

Checks:
    - Dataset information
    - Null values
    - Duplicate rows
    - transaction_id format
    - applicant_id matching
    - Platform values
    - Transaction dates
    - Transaction types
    - Amount parsing
    - Direction values
    - Counterparty category
    - Geolocation
    - Noise columns

"""

import pandas as pd
from pathlib import Path
import re


# ==========================================================================
# Configuration
# ==========================================================================

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "cleaned_datasets"
    / "mobile_money_transactions_ver1.parquet"
)

MAX_PRINT_ROWS = 10


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

    if drop_duplicates:

        df = df.drop_duplicates()

        print()

        print("Duplicate rows removed.")

        print(f"Remaining rows : {len(df):,}")

    return df


# ==========================================================================
# transaction_id Validation
# ==========================================================================

def check_transaction_id(
    df,
    drop_invalid_format=False,
    drop_mismatched_applicant=False,
    drop_duplicate_transaction_ids=False
):

    print("\n")
    print("=" * 80)
    print("TRANSACTION ID VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------
    # Format Check
    # ------------------------------------------------------

    print("\nChecking transaction_id format...")

    valid_format = (
        df["transaction_id"]
        .astype(str)
        .str.match(r"^TX-AP-\d{6}-\d{4}$")
    )

    invalid = ~valid_format

    print(f"Invalid transaction IDs : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                ["transaction_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid_format:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid transaction IDs removed.")

    # ------------------------------------------------------
    # applicant_id match
    # ------------------------------------------------------

    print("\nChecking applicant_id consistency...")

    extracted = (
        df["transaction_id"]
        .str.extract(r"(AP-\d{6})")[0]
    )

    mismatch = (
        extracted
        !=
        df["applicant_id"]
    )

    print(f"Mismatched IDs : {mismatch.sum():,}")

    if mismatch.sum():

        print()

        print(
            df.loc[
                mismatch,
                [
                    "transaction_id",
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
    # Duplicate transaction IDs
    # ------------------------------------------------------

    print("\nChecking duplicate transaction_id values...")

    duplicate_ids = (
        df["transaction_id"]
        .duplicated()
    )

    print(f"Duplicate IDs : {duplicate_ids.sum():,}")

    if duplicate_ids.sum():

        print()

        print(
            df.loc[
                duplicate_ids,
                ["transaction_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_duplicate_transaction_ids:

        df = df.loc[
            ~duplicate_ids
        ].copy()

        print("\nDuplicate transaction IDs removed.")

    return df

# ==========================================================================
# Platform Check
# ==========================================================================

def check_platform(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("PLATFORM CHECK")
    print("=" * 80)

    expected = {
        "esewa",
        "khalti"
    }

    distribution = (
        df["platform"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Platform Distribution:\n")
    print(distribution)

    invalid = ~df["platform"].isin(expected)

    print(f"\nInvalid platform values : {invalid.sum():,}")
    print(f"Percentage              : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nUnexpected values:\n")

        print(
            df.loc[
                invalid,
                "platform"
            ].value_counts(dropna=False)
        )

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "transaction_id",
                    "platform"
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
# Transaction Date Check
# ==========================================================================

def check_transaction_date(
    df,
    latest_valid=None, 
# making the latest_valid None and later putting the current system time in it,
# we are doing this since default parameters are evaluated once the function is called, instead of each time it's called 
    drop_invalid=False
):

# putting the time stamp now due to the above listed reasons
    if latest_valid is None:
        latest_valid = pd.Timestamp.now()

    print("\n")
    print("=" * 80)
    print("TRANSACTION DATE CHECK")
    print("=" * 80)

    parsed_dates = pd.to_datetime(
        df["transaction_date"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    latest_valid = pd.Timestamp(latest_valid)

    invalid = (
        parsed_dates.isna()
        |
        (parsed_dates > latest_valid)
    )

    print(f"Impossible timestamps : {invalid.sum():,}")
    print(f"Percentage            : {invalid.mean()*100:.2f}%")

    print("\nDate Range")

    print(f"Earliest : {parsed_dates.min()}")
    print(f"Latest   : {parsed_dates.max()}")

    if invalid.sum():

        print("\nExample invalid rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "transaction_id",
                    "transaction_date"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nImpossible timestamps removed.")

    return df


# ==========================================================================
# Transaction Type Check
# ==========================================================================

def check_transaction_type(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("TRANSACTION TYPE CHECK")
    print("=" * 80)

    expected = {

        "merchant_payment",
        "p2p_transfer",
        "utility_payment",
        "qr_payment",
        "remittance_receipt",
        "wallet_topup",
        "wallet_withdrawal",
        "loan_repayment"

    }

    distribution = (
        df["transaction_type"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Transaction Type Distribution:\n")
    print(distribution)

    invalid = (
        ~df["transaction_type"]
        .isin(expected)
    )

    print(f"\nInvalid values : {invalid.sum():,}")
    print(f"Percentage     : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nUnexpected values:\n")

        print(
            df.loc[
                invalid,
                "transaction_type"
            ].value_counts(dropna=False)
        )

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "transaction_id",
                    "transaction_type"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid transaction types removed.")

    return df


# ==========================================================================
# Amount Check
# ==========================================================================

def check_amount(
    df,
    drop_unparseable=False,
    drop_non_positive=False
):

    print("\n")
    print("=" * 80)
    print("AMOUNT CHECK")
    print("=" * 80)

    parsed_amount = pd.to_numeric(
        df["amount_nrs"]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

    invalid = parsed_amount.isna()

    print("Parsing Amounts")

    print(f"Unable to parse : {invalid.sum():,}")
    print(f"Percentage      : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "transaction_id",
                    "amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    non_positive = parsed_amount <= 0

    print("\nChecking for zero or negative amounts")

    print(f"Rows : {non_positive.sum():,}")
    print(f"Percentage : {non_positive.mean()*100:.2f}%")

    if non_positive.sum():

        print()

        print(
            df.loc[
                non_positive,
                [
                    "transaction_id",
                    "amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nAmount Statistics\n")

    print(parsed_amount.describe())

    if drop_unparseable:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nRows with invalid amounts removed.")

    if drop_non_positive:

        df = df.loc[
            ~non_positive
        ].copy()

        print("\nRows with non-positive amounts removed.")

    return df

# ==========================================================================
# Direction Check
# ==========================================================================

def check_direction(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("DIRECTION CHECK")
    print("=" * 80)

    expected = {
        "credit",
        "debit"
    }

    distribution = (
        df["direction"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Direction Distribution:\n")
    print(distribution)

    invalid = ~df["direction"].isin(expected)

    print(f"\nInvalid values : {invalid.sum():,}")
    print(f"Percentage     : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nUnexpected values:\n")

        print(
            df.loc[
                invalid,
                "direction"
            ].value_counts(dropna=False)
        )

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "transaction_id",
                    "direction"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid direction rows removed.")

    return df


# ==========================================================================
# Counterparty Category Check
# ==========================================================================

def check_counterparty_category(
    df,
    drop_null=False,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("COUNTERPARTY CATEGORY CHECK")
    print("=" * 80)

    expected = {

        "grocery",
        "utility",
        "telecom",
        "agriculture_input",
        "medical",
        "transport",
        "financial_services",
        "remittance_agent",
        "education",
        "restaurant"

    }

    distribution = (
        df["counterparty_category"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Counterparty Category Distribution:\n")
    print(distribution)

    # ---------------- NULL CHECK ----------------

    null_rows = df["counterparty_category"].isna()

    print("\nNULL Values")

    print(f"Count      : {null_rows.sum():,}")
    print(f"Percentage : {null_rows.mean()*100:.2f}%")

    if null_rows.sum():

        print("\nExample NULL rows:\n")

        print(
            df.loc[
                null_rows,
                [
                    "transaction_id",
                    "counterparty_category"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ---------------- INVALID CHECK ----------------

    invalid = (

        df["counterparty_category"].notna()

        &

        ~df["counterparty_category"].isin(expected)

    )

    print("\nInvalid Values")

    print(f"Count      : {invalid.sum():,}")
    print(f"Percentage : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nUnexpected values:\n")

        print(
            df.loc[
                invalid,
                "counterparty_category"
            ].value_counts(dropna=False)
        )

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "transaction_id",
                    "counterparty_category"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_null:

        df = df.loc[
            ~null_rows
        ].copy()

        print("\nNULL rows removed.")

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Geolocation Check
# ==========================================================================

def check_geolocation(df):

    print("\n")
    print("=" * 80)
    print("GEOLOCATION CHECK")
    print("=" * 80)

    distribution = (
        df["geolocation_district"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print(distribution)


# ==========================================================================
# Noise Columns Check
# ==========================================================================

def check_noise_columns(df):

    print("\n")
    print("=" * 80)
    print("NOISE COLUMN SUMMARY")
    print("=" * 80)

    columns = [

        "_noise_anomaly_flag",
        "_noise_anomaly_type",
        "_noise_null_party",
        "_noise_amount_string"

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
        drop_duplicates=False
    )

    df = check_transaction_id(
        df,
        drop_invalid_format=True,
        drop_mismatched_applicant=False,
        drop_duplicate_transaction_ids=False
    )

    df = check_platform(
        df,
        drop_invalid=False
    )

    df = check_transaction_date(
        df,
        latest_valid="2025-12-31 23:59:59",
        drop_invalid=True
    )

    df = check_transaction_type(
        df,
        drop_invalid=False
    )

    df = check_amount(
        df,
        drop_unparseable=False,
        drop_non_positive=False
    )

    df = check_direction(
        df,
        drop_invalid=False
    )

    df = check_counterparty_category(
        df,
        drop_null=True,
        drop_invalid=False
    )

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
        r"mobile_money_transactions_ver(\d+)\.parquet"
    )

    versions = []

    for file in output_folder.glob("mobile_money_transactions_ver*.parquet"):

        match = pattern.fullmatch(file.name)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1

    output_file = (
        output_folder
        / f"mobile_money_transactions_ver{next_version}.parquet"
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
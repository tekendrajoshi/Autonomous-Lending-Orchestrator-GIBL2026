"""
===========================================================================
Utility Payments Validator
===========================================================================

Checks:
    - Dataset information
    - Null values
    - Duplicate rows
    - payment_id format
    - applicant_id matching
    - Address fields (province / district / municipality / ward)
    - Utility type / provider consistency
    - Service number format
    - Billing period (BS / AD)
    - Bill amount
    - Units consumed
    - Due date
    - Payment date / payment method / days late (unpaid-consistency)
    - Cumulative on-time rate
    - Outstanding arrears
    - Noise columns

NOTE ON CROSS-PARQUET CHECKS
-----------------------------
This script deliberately only validates utility_payments.parquet in
isolation. The following checks CANNOT be done here and would need a join
against other datasets (flagged inline with "CROSS-PARQUET CHECK NEEDED"):

    1. applicant_id existence        -> must exist in applicant_profiles.parquet
    2. province_en / district_en /
       municipality_en / ward_no     -> should match applicant_profiles.parquet
                                         address fields for the same applicant
    3. billing_period_bs <-> _ad     -> a proper BS<->AD calendar conversion
                                         table would be needed to fully verify
                                         these two columns agree; this script
                                         only checks each column's own format
    4. cumulative_on_time_rate       -> to fully verify this is a correct
                                         running calculation, it should be
                                         recomputed from the full payment
                                         history per applicant/utility_type
                                         ordered by billing_period, rather
                                         than trusted as stored

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
    / "utility_payments.parquet"
)

MAX_PRINT_ROWS = 10

EXPECTED_UTILITY_TYPES = {
    "electricity",
    "mobile"
}

EXPECTED_PROVIDERS = {
    "NEA",
    "Ncell"
}

# utility_type -> expected provider
UTILITY_TYPE_PROVIDER_MAP = {
    "electricity": "NEA",
    "mobile": "Ncell"
}

EXPECTED_PAYMENT_METHODS = {
    "esewa",
    "khalti",
    "bank"
}

ELECTRICITY_RATE_PER_UNIT = 7.30


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

    print("\nNOTE: ward_no, units_consumed, payment_date_ad, payment_method,")
    print("and days_late are expected to contain NULLs by design (see data")
    print("dictionary). All other columns should be fully populated.")

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
                        "payment_id",
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
# payment_id Validation
# ==========================================================================

def check_payment_id(
    df,
    drop_invalid_format=False,
    drop_mismatched_applicant=False,
    drop_duplicate_ids=False
):

    print("\n")
    print("=" * 80)
    print("PAYMENT ID VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------
    # Format Check
    # ------------------------------------------------------

    print("\nChecking payment_id format...")

    valid_format = (
        df["payment_id"]
        .astype(str)
        .str.match(r"^UTIL-AP-\d{6}-(ELE|MOB)-\d{2}$", case=False)
    )

    invalid = ~valid_format

    print(f"Invalid payment IDs : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                ["payment_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid_format:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid payment IDs removed.")

    # ------------------------------------------------------
    # applicant_id match (format-level only)
    # ------------------------------------------------------

    print("\nChecking applicant_id consistency with payment_id...")
    print("NOTE: this only checks internal consistency between the two")
    print("columns in this file. Whether applicant_id actually exists is a")
    print("CROSS-PARQUET CHECK NEEDED against applicant_profiles.parquet.")

    extracted = (
        df["payment_id"]
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
                    "payment_id",
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
    # payment_id suffix vs utility_type consistency
    # ------------------------------------------------------

    print("\nChecking payment_id suffix (ELE/MOB) vs utility_type...")

    suffix = (
        df["payment_id"]
        .str.extract(r"-(ELE|MOB)-", flags=re.IGNORECASE)[0]
        .str.upper()
    )

    expected_suffix = df["utility_type"].map({
        "electricity": "ELE",
        "mobile": "MOB"
    })

    suffix_mismatch = (
        suffix.notna()
        & expected_suffix.notna()
        & (suffix != expected_suffix)
    )

    print(f"Suffix / utility_type mismatches : {suffix_mismatch.sum():,}")

    if suffix_mismatch.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                suffix_mismatch,
                [
                    "payment_id",
                    "utility_type"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Duplicate payment IDs
    # ------------------------------------------------------

    print("\nChecking duplicate payment_id values...")

    duplicate_ids = (
        df["payment_id"]
        .duplicated()
    )

    print(f"Duplicate IDs : {duplicate_ids.sum():,}")

    if duplicate_ids.sum():

        print()

        print(
            df.loc[
                duplicate_ids,
                ["payment_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_duplicate_ids:

        df = df.loc[
            ~duplicate_ids
        ].copy()

        print("\nDuplicate payment IDs removed.")

    return df


# ==========================================================================
# Address Field Check
# ==========================================================================

def check_address_fields(df):

    print("\n")
    print("=" * 80)
    print("ADDRESS FIELD CHECK")
    print("=" * 80)

    print("Checking province_en / district_en / municipality_en / ward_no")
    print("for NULL or blank values only.")

    print("\nCROSS-PARQUET CHECK NEEDED: compare these fields against")
    print("applicant_profiles.parquet to confirm they match the address on")
    print("file for the same applicant_id.")

    for column in [
        "province_en",
        "district_en",
        "municipality_en"
    ]:

        blank = (
            df[column].isna()
            |
            (df[column].astype(str).str.strip() == "")
        )

        print(f"\n{column}")
        print(f"  Blank / NULL rows : {blank.sum():,}")
        print(f"  Percentage        : {blank.mean()*100:.2f}%")

    print("\nward_no")

    ward_null = df["ward_no"].isna()

    print(f"  NULL rows  : {ward_null.sum():,}")
    print(f"  Percentage : {ward_null.mean()*100:.2f}%")

    non_null_ward = pd.to_numeric(
        df.loc[~ward_null, "ward_no"],
        errors="coerce"
    )

    non_positive_ward = non_null_ward < 1

    print(f"  Non-positive ward numbers (excl. NULL) : {non_positive_ward.sum():,}")


# ==========================================================================
# Utility Type / Provider Check
# ==========================================================================

def check_utility_type_and_provider(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("UTILITY TYPE / PROVIDER CHECK")
    print("=" * 80)

    type_distribution = (
        df["utility_type"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    type_distribution["Percentage"] = (
        type_distribution["Count"] / len(df) * 100
    ).round(2)

    print("Utility Type Distribution:\n")
    print(type_distribution)

    provider_distribution = (
        df["provider"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    provider_distribution["Percentage"] = (
        provider_distribution["Count"] / len(df) * 100
    ).round(2)

    print("\nProvider Distribution:\n")
    print(provider_distribution)

    invalid_type = ~df["utility_type"].isin(EXPECTED_UTILITY_TYPES)
    invalid_provider = ~df["provider"].isin(EXPECTED_PROVIDERS)

    print(f"\nInvalid utility_type values : {invalid_type.sum():,}")
    print(f"Invalid provider values     : {invalid_provider.sum():,}")

    # ------------------------------------------------------
    # Cross-column consistency: utility_type <-> provider
    # ------------------------------------------------------

    print("\nChecking utility_type <-> provider consistency...")

    expected_provider = df["utility_type"].map(UTILITY_TYPE_PROVIDER_MAP)

    provider_mismatch = (
        expected_provider.notna()
        & (df["provider"] != expected_provider)
    )

    print(f"Mismatches : {provider_mismatch.sum():,}")

    if provider_mismatch.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                provider_mismatch,
                [
                    "payment_id",
                    "utility_type",
                    "provider"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    invalid = invalid_type | invalid_provider | provider_mismatch

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Service Number Check
# ==========================================================================

def check_service_number(df):

    print("\n")
    print("=" * 80)
    print("SERVICE NUMBER CHECK")
    print("=" * 80)

    valid_format = (
        df["service_number"]
        .astype(str)
        .str.match(r"^[A-Za-z]{2,4}-\d{5}$")
    )

    invalid = ~valid_format

    print(f"Invalid service_number values : {invalid.sum():,}")
    print(f"Percentage                    : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "payment_id",
                    "service_number"
                ]
            ].head(MAX_PRINT_ROWS)
        )


# ==========================================================================
# Billing Period Check
# ==========================================================================

def check_billing_period(df):

    print("\n")
    print("=" * 80)
    print("BILLING PERIOD CHECK")
    print("=" * 80)

    valid_bs = (
        df["billing_period_bs"]
        .astype(str)
        .str.match(r"^\d{4}-\d{2}$")
    )

    valid_ad = (
        df["billing_period_ad"]
        .astype(str)
        .str.match(r"^\d{4}-\d{2}$")
    )

    print(f"Invalid billing_period_bs format : {(~valid_bs).sum():,}")
    print(f"Invalid billing_period_ad format : {(~valid_ad).sum():,}")

    if (~valid_bs).sum():

        print("\nExample invalid BS rows:\n")

        print(
            df.loc[
                ~valid_bs,
                ["payment_id", "billing_period_bs"]
            ].head(MAX_PRINT_ROWS)
        )

    if (~valid_ad).sum():

        print("\nExample invalid AD rows:\n")

        print(
            df.loc[
                ~valid_ad,
                ["payment_id", "billing_period_ad"]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nAD Billing Period Distribution:\n")

    print(
        df["billing_period_ad"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nCROSS-PARQUET CHECK NEEDED (or a BS<->AD conversion table):")
    print("this script only checks each column's own format — it does not")
    print("verify that billing_period_bs and billing_period_ad refer to the")
    print("same actual calendar month.")


# ==========================================================================
# Bill Amount Check
# ==========================================================================

def check_bill_amount(
    df,
    clip_negatives=False,
    drop_negatives=False,
    drop_outliers=False
):

    print("\n")
    print("=" * 80)
    print("BILL AMOUNT CHECK")
    print("=" * 80)

    # ------------------------------------------------------
    # Parse Amount
    # ------------------------------------------------------

    amount = pd.to_numeric(
        df["bill_amount_nrs"],
        errors="coerce"
    )

    invalid = amount.isna()

    print(f"Unable to parse : {invalid.sum():,}")
    print(f"Percentage      : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample invalid rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "payment_id",
                    "bill_amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Negative Values
    # ------------------------------------------------------

    negative = amount < 0

    print("\nNegative bill amounts")
    print(f"Rows       : {negative.sum():,}")
    print(f"Percentage : {negative.mean()*100:.2f}%")

    if negative.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                negative,
                [
                    "payment_id",
                    "bill_amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Compare against injected noise
    # ------------------------------------------------------

    if "_noise_negative_bill" in df.columns:

        flagged = (
            df["_noise_negative_bill"]
            == True
        )

        mismatch = (
            flagged
            !=
            negative
        )

        print(
            f"\nMismatch with _noise_negative_bill : "
            f"{mismatch.sum():,}"
        )

    # ------------------------------------------------------
    # Typical Range
    # ------------------------------------------------------

    out_of_typical_range = (

        amount.notna()

        &

        (amount >= 0)

        &

        (
            (amount < 150)

            |

            (amount > 5000)
        )

    )

    print("\nOutside Typical Range (150–5000)")
    print(f"Rows       : {out_of_typical_range.sum():,}")
    print(f"Percentage : {out_of_typical_range.mean()*100:.2f}%")

    if out_of_typical_range.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                out_of_typical_range,
                [
                    "payment_id",
                    "bill_amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    print("\nBill Amount Statistics\n")

    print(amount.describe())

    # ------------------------------------------------------
    # Electricity Correlation
    # ------------------------------------------------------

    print(
        "\nChecking electricity bills "
        f"(units × {ELECTRICITY_RATE_PER_UNIT})..."
    )

    electricity = (
        df["utility_type"]
        == "electricity"
    )

    units = pd.to_numeric(
        df.loc[
            electricity,
            "units_consumed"
        ],
        errors="coerce"
    )

    expected_bill = (
        units
        *
        ELECTRICITY_RATE_PER_UNIT
    )

    actual_bill = pd.to_numeric(
        df.loc[
            electricity,
            "bill_amount_nrs"
        ],
        errors="coerce"
    )

    diff_pct = (

        (actual_bill - expected_bill).abs()

        /

        expected_bill.replace(
            0,
            pd.NA
        )

    ) * 100

    mismatched = (
        diff_pct > 5
    )

    print(
        f"Electricity rows with >5% discrepancy : "
        f"{mismatched.sum():,}"
    )

    if mismatched.sum():

        print("\nExample rows:\n")

        print(

            df.loc[
                electricity,
                [
                    "payment_id",
                    "units_consumed",
                    "bill_amount_nrs"
                ]
            ]
            .loc[mismatched]
            .head(MAX_PRINT_ROWS)

        )

    # ------------------------------------------------------
    # Negative Handling
    # ------------------------------------------------------

    if clip_negatives:

        print("\nClipping negative bill amounts to 0...")

        amount = amount.clip(lower=0)

        df["bill_amount_nrs"] = amount

        print(
            f"Values clipped : "
            f"{negative.sum():,}"
        )

    elif drop_negatives:

        before = len(df)

        df = df.loc[
            ~negative
        ].copy()

        amount = pd.to_numeric(
            df["bill_amount_nrs"],
            errors="coerce"
        )

        print(
            f"\nDropped "
            f"{before-len(df):,} "
            "rows with negative bill amounts."
        )

    # ------------------------------------------------------
    # Outlier Detection (IQR)
    # ------------------------------------------------------

    q1 = amount.quantile(0.25)
    q3 = amount.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = (

        (amount < lower_bound)

        |

        (amount > upper_bound)

    )

    print("\nOutlier Detection (IQR)")

    print(f"Lower Bound : {lower_bound:.2f}")
    print(f"Upper Bound : {upper_bound:.2f}")

    print(f"Rows        : {outliers.sum():,}")
    print(f"Percentage  : {outliers.mean()*100:.2f}%")

    if outliers.sum():

        print("\nExample outlier rows:\n")

        print(
            df.loc[
                outliers,
                [
                    "payment_id",
                    "bill_amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_outliers:

        before = len(df)

        df = df.loc[
            ~outliers
        ].copy()

        print(
            f"\nDropped "
            f"{before-len(df):,} "
            "outlier rows."
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check total payment amount with
    # billing history if available.
    #
    # Compare monthly bill trends for sudden
    # spikes or suspicious decreases.
    #
    # Compare with applicant income to
    # identify unusually high utility bills.
    #

    return df


def check_units_consumed(
    df,
    clip_negatives=False,
    drop_negatives=False,
    drop_outliers=False
):

    print("\n")
    print("=" * 80)
    print("UNITS CONSUMED CHECK")
    print("=" * 80)

    # ------------------------------------------------------
    # Parse Units
    # ------------------------------------------------------

    units = pd.to_numeric(
        df["units_consumed"],
        errors="coerce"
    )

    invalid = units.isna()

    print(f"Unable to parse : {invalid.sum():,}")
    print(f"Percentage      : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "payment_id",
                    "utility_type",
                    "units_consumed"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Negative Values
    # ------------------------------------------------------

    negative = units < 0

    print("\nNegative units consumed")
    print(f"Rows       : {negative.sum():,}")
    print(f"Percentage : {negative.mean()*100:.2f}%")

    if negative.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                negative,
                [
                    "payment_id",
                    "utility_type",
                    "units_consumed"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Zero Values
    # ------------------------------------------------------

    zero = units == 0

    print("\nZero units consumed")
    print(f"Rows       : {zero.sum():,}")
    print(f"Percentage : {zero.mean()*100:.2f}%")

    if zero.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                zero,
                [
                    "payment_id",
                    "utility_type",
                    "units_consumed"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Typical Range
    # ------------------------------------------------------

    typical = (

        units.notna()

        &

        (units >= 0)

        &

        (
            (units < 10)

            |

            (units > 1000)
        )

    )

    print("\nOutside Typical Range (10–1000)")
    print(f"Rows       : {typical.sum():,}")
    print(f"Percentage : {typical.mean()*100:.2f}%")

    if typical.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                typical,
                [
                    "payment_id",
                    "utility_type",
                    "units_consumed"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    print("\nUnits Consumed Statistics\n")

    print(units.describe())

    # ------------------------------------------------------
    # Negative Handling
    # ------------------------------------------------------

    if clip_negatives and drop_negatives:

        raise ValueError(
            "Only one of clip_negatives or "
            "drop_negatives can be True."
        )

    if clip_negatives:

        print("\nClipping negative units to 0...")

        units = units.clip(lower=0)

        df["units_consumed"] = units

        print(
            f"Values clipped : "
            f"{negative.sum():,}"
        )

    elif drop_negatives:

        before = len(df)

        df = df.loc[
            ~negative
        ].copy()

        units = pd.to_numeric(
            df["units_consumed"],
            errors="coerce"
        )

        print(
            f"\nDropped "
            f"{before-len(df):,} "
            "rows with negative units."
        )

    # ------------------------------------------------------
    # Outlier Detection (IQR)
    # ------------------------------------------------------

    q1 = units.quantile(0.25)
    q3 = units.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = (

        (units < lower_bound)

        |

        (units > upper_bound)

    )

    print("\nOutlier Detection (IQR)")

    print(f"Lower Bound : {lower_bound:.2f}")
    print(f"Upper Bound : {upper_bound:.2f}")

    print(f"Rows        : {outliers.sum():,}")
    print(f"Percentage  : {outliers.mean()*100:.2f}%")

    if outliers.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                outliers,
                [
                    "payment_id",
                    "utility_type",
                    "units_consumed"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_outliers:

        before = len(df)

        df = df.loc[
            ~outliers
        ].copy()

        print(
            f"\nDropped "
            f"{before-len(df):,} "
            "outlier rows."
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check units_consumed with
    # bill_amount_nrs.
    #
    # Electricity:
    # bill ≈ units × ELECTRICITY_RATE_PER_UNIT
    #
    # Water:
    # compare against typical household usage.
    #
    # Extremely high units with low bill,
    # or vice versa, may indicate injected noise.
    #

    return df


# ==========================================================================
# Due Date Check
# ==========================================================================

def check_due_date(df):

    print("\n")
    print("=" * 80)
    print("DUE DATE CHECK")
    print("=" * 80)

    due_dates = pd.to_datetime(
        df["due_date_ad"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    invalid = due_dates.isna()

    print(f"Unable to parse due_date_ad : {invalid.sum():,}")

    not_28th = (
        due_dates.notna()
        &
        (due_dates.dt.day != 28)
    )

    print(f"\nDue dates not on the 28th")
    print(f"Rows       : {not_28th.sum():,}")
    print(f"Percentage : {not_28th.mean()*100:.2f}%")

    if not_28th.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                not_28th,
                [
                    "payment_id",
                    "due_date_ad"
                ]
            ].head(MAX_PRINT_ROWS)
        )


# ==========================================================================
# Payment Status Check
# (payment_date_ad / payment_method / days_late / outstanding_arrears_nrs)
# ==========================================================================

def check_payment_status(
    df,
    drop_invalid_method=False
):

    print("\n")
    print("=" * 80)
    print("PAYMENT STATUS CHECK")
    print("=" * 80)

    unpaid = df["payment_date_ad"].isna()

    print(f"Unpaid bills (payment_date_ad is NULL) : {unpaid.sum():,}")
    print(f"Percentage                              : {unpaid.mean()*100:.2f}%")

    if "_noise_forced_unpaid" in df.columns:

        forced = df["_noise_forced_unpaid"] == True

        print(f"\nRows flagged as forced-unpaid noise : {forced.sum():,}")

    # ------------------------------------------------------
    # payment_method should be NULL iff unpaid
    # ------------------------------------------------------

    print("\nChecking payment_method is NULL iff unpaid...")

    method_null = df["payment_method"].isna()

    method_mismatch = method_null != unpaid

    print(f"Mismatches : {method_mismatch.sum():,}")

    if method_mismatch.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                method_mismatch,
                [
                    "payment_id",
                    "payment_date_ad",
                    "payment_method"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    invalid_method = (
        df["payment_method"].notna()
        &
        ~df["payment_method"].isin(EXPECTED_PAYMENT_METHODS)
    )

    print(f"\nInvalid (non-NULL) payment_method values : {invalid_method.sum():,}")

    if invalid_method.sum():

        print("\nUnexpected values:\n")

        print(
            df.loc[
                invalid_method,
                "payment_method"
            ].value_counts(dropna=False)
        )

    # ------------------------------------------------------
    # days_late should be NULL iff unpaid, and consistent with
    # payment_date_ad - due_date_ad
    # ------------------------------------------------------

    print("\nChecking days_late is NULL iff unpaid...")

    days_late_null = df["days_late"].isna()

    days_late_mismatch = days_late_null != unpaid

    print(f"Mismatches : {days_late_mismatch.sum():,}")

    print("\nChecking days_late = payment_date_ad - due_date_ad...")

    payment_dates = pd.to_datetime(
        df["payment_date_ad"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    due_dates = pd.to_datetime(
        df["due_date_ad"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    computed_days_late = (payment_dates - due_dates).dt.days

    reported_days_late = pd.to_numeric(
        df["days_late"],
        errors="coerce"
    )

    both_present = payment_dates.notna() & reported_days_late.notna()

    days_late_diff = (
        (computed_days_late - reported_days_late).abs()
    )

    days_late_wrong = both_present & (days_late_diff > 0)

    print(f"Rows where days_late doesn't match date arithmetic : "
          f"{days_late_wrong.sum():,}")

    if days_late_wrong.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                days_late_wrong,
                [
                    "payment_id",
                    "due_date_ad",
                    "payment_date_ad",
                    "days_late"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # outstanding_arrears_nrs should be non-zero only if unpaid
    # ------------------------------------------------------

    print("\nChecking outstanding_arrears_nrs consistency with unpaid status...")

    arrears = pd.to_numeric(
        df["outstanding_arrears_nrs"],
        errors="coerce"
    )

    negative_arrears = arrears < 0

    print(f"Negative arrears values : {negative_arrears.sum():,}")

    arrears_but_paid = (~unpaid) & (arrears > 0)

    print(f"Rows paid but with non-zero arrears : {arrears_but_paid.sum():,}")

    unpaid_but_zero_arrears = unpaid & (arrears == 0)

    print(f"Rows unpaid but with zero arrears   : {unpaid_but_zero_arrears.sum():,}")

    if arrears_but_paid.sum():

        print("\nExample rows (paid but non-zero arrears):\n")

        print(
            df.loc[
                arrears_but_paid,
                [
                    "payment_id",
                    "payment_date_ad",
                    "outstanding_arrears_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid_method:

        df = df.loc[
            ~invalid_method
        ].copy()

        print("\nRows with invalid payment_method removed.")

    return df


# ==========================================================================
# Cumulative On-Time Rate Check
# ==========================================================================

def check_cumulative_on_time_rate(df):

    print("\n")
    print("=" * 80)
    print("CUMULATIVE ON-TIME RATE CHECK")
    print("=" * 80)

    rate = pd.to_numeric(
        df["cumulative_on_time_rate"],
        errors="coerce"
    )

    invalid = rate.isna()

    print(f"Unable to parse : {invalid.sum():,}")

    out_of_range = (
        rate.notna()
        &
        ((rate < 0.30) | (rate > 1.0))
    )

    print(f"\nOut-of-range values (expected 0.30–1.0)")
    print(f"Rows       : {out_of_range.sum():,}")
    print(f"Percentage : {out_of_range.mean()*100:.2f}%")

    if out_of_range.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                out_of_range,
                [
                    "payment_id",
                    "cumulative_on_time_rate"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nRate Statistics\n")

    print(rate.describe())

    print("\nCROSS-PARQUET CHECK NEEDED / SEQUENCE CHECK NEEDED: fully")
    print("validating this column means recomputing the running on-time")
    print("proportion per applicant_id + utility_type ordered by")
    print("billing_period, rather than trusting the stored value here.")


# ==========================================================================
# Noise Columns Check
# ==========================================================================

def check_noise_columns(df):

    print("\n")
    print("=" * 80)
    print("NOISE COLUMN SUMMARY")
    print("=" * 80)

    columns = [

        "_noise_forced_unpaid",
        "_noise_negative_bill",
        "_noise_zero_units",
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

    df = check_payment_id(
        df,
        drop_invalid_format=True,
        drop_mismatched_applicant=False,
        drop_duplicate_ids=True
    )

    check_address_fields(df)

    df = check_utility_type_and_provider(
        df,
        drop_invalid=False
    )

    check_service_number(df)

    check_billing_period(df)

    df = check_bill_amount(
        df,
        clip_negatives=True,
        drop_outliers=False
    )

    check_units_consumed(
        df,
        clip_negatives=True,
        drop_outliers=False
    )

    check_due_date(df)

    df = check_payment_status(
        df,
        drop_invalid_method=False
    )

    check_cumulative_on_time_rate(df)

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
        r"utility_payments_ver(\d+)\.parquet"
    )

    versions = []

    for file in output_folder.glob("utility_payments_ver*.parquet"):

        match = pattern.fullmatch(file.name)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1

    output_file = (
        output_folder
        / f"utility_payments_ver{next_version}.parquet"
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
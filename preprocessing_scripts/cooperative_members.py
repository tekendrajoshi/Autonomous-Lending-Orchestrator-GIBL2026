"""
===========================================================================
Cooperative Members Validator
===========================================================================

Checks:
    - Dataset information
    - Null values
    - Duplicate rows
    - member_id format
    - applicant_id matching
    - cooperative_id format
    - Cooperative name fields
    - Cooperative type
    - Address fields (province / district / municipality)
    - Membership year
    - Share count / share value / total share value
    - Last annual dividend
    - Outstanding loan
    - Coop loan repayment status
    - Membership status

COLUMN-EXISTENCE GUARD
-----------------------
Every check function verifies its required columns are present in the
dataframe before running. If a required column is missing, the check is
skipped and a message is printed instead of raising a KeyError.

NOTE ON CROSS-TABLE CHECKS
----------------------------
This script only validates cooperative_members.csv in isolation. The
following checks CANNOT be done here and would need a join against other
tables (flagged inline with "CROSS-TABLE CHECK NEEDED"):

    1. applicant_id existence          -> must exist in applicant_profiles.csv
    2. "Ghost membership" detection    -> applicants with
                                           cooperative_member = True in
                                           applicant_profiles.csv but NO row
                                           here (the inverse direction of
                                           this same noise pattern — i.e.
                                           confirming every row here also has
                                           cooperative_member = True on the
                                           profile — can only be checked with
                                           a join)
    3. province_en / district_en /
       municipality_en                 -> should match the member's
                                           applicant_profiles.csv address
    4. cooperative_id referenced by
       cooperative_sales.csv           -> every sale's cooperative_id should
                                           resolve to a row here

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
    / "cooperative_members.csv"
)

MAX_PRINT_ROWS = 10

EXPECTED_COOPERATIVE_TYPES = {
    "agricultural", "dairy", "savings_credit", "vegetable", "coffee_tea"
}

EXPECTED_PROVINCES = {
    "Koshi", "Madhesh", "Bagmati", "Gandaki",
    "Lumbini", "Karnali", "Sudurpashchim"
}

EXPECTED_REPAYMENT_STATUSES = {"none", "current", "overdue"}

EXPECTED_MEMBERSHIP_STATUSES = {"active", "inactive", "suspended"}


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

    print("\nNOTE: no column in this table is documented as nullable in the")
    print("data dictionary. Any NULLs found here are unexpected.")

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

    print(f"Duplicate Rows : {count:,}")
    print(f"Percentage     : {count / len(df) * 100:.2f}%")

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
# member_id / applicant_id Validation
# ==========================================================================

def check_ids(
    df,
    drop_invalid_format=False,
    drop_mismatched_applicant=False,
    drop_duplicate_ids=False
):

    print("\n")
    print("=" * 80)
    print("MEMBER ID / APPLICANT ID VALIDATION")
    print("=" * 80)

    if require_columns(df, ["member_id"], "MEMBER ID / APPLICANT ID VALIDATION"):
        return df

    valid_format = (
        df["member_id"]
        .astype(str)
        .str.match(r"^CMEM-AP-\d{6}$")
    )

    invalid = ~valid_format

    print(f"Invalid member_id format : {invalid.sum():,}")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                ["member_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid_format:

        df = df.loc[~invalid].copy()

        print("\nInvalid member IDs removed.")

    if "applicant_id" in df.columns:

        print("\nChecking applicant_id consistency with member_id...")

        extracted = (
            df["member_id"]
            .str.extract(r"(AP-\d{6})")[0]
        )

        mismatch = extracted != df["applicant_id"]

        print(f"Mismatched IDs : {mismatch.sum():,}")

        if mismatch.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    mismatch,
                    ["member_id", "applicant_id"]
                ].head(MAX_PRINT_ROWS)
            )

        if drop_mismatched_applicant:

            df = df.loc[~mismatch].copy()

            print("\nMismatched rows removed.")

        print("\nCROSS-TABLE CHECK NEEDED: confirm every applicant_id here")
        print("exists in applicant_profiles.csv with cooperative_member = ")
        print("True.")

    else:

        print("\nSKIPPING applicant_id consistency check — column missing.")

    print("\nChecking duplicate member_id values (should be a primary key)...")

    duplicate_ids = df["member_id"].duplicated()

    print(f"Duplicate member IDs : {duplicate_ids.sum():,}")

    if drop_duplicate_ids:

        df = df.loc[~duplicate_ids].copy()

        print("\nDuplicate member IDs removed.")

    return df


# ==========================================================================
# cooperative_id Check
# ==========================================================================

def check_cooperative_id(df):

    print("\n")
    print("=" * 80)
    print("COOPERATIVE ID CHECK")
    print("=" * 80)

    if require_columns(df, ["cooperative_id"], "COOPERATIVE ID CHECK"):
        return

    valid_format = (
        df["cooperative_id"]
        .astype(str)
        .str.match(r"^COOP-[A-Za-z]{2,4}-\d{4}$")
    )

    invalid = ~valid_format

    print(f"Invalid cooperative_id format : {invalid.sum():,}")

    if invalid.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                invalid,
                ["member_id", "cooperative_id"]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nNumber of unique cooperatives represented:")

    print(df["cooperative_id"].nunique())

    print("\nCROSS-TABLE CHECK NEEDED: every cooperative_id referenced in")
    print("cooperative_sales.csv should resolve to a row here.")


# ==========================================================================
# Cooperative Name Check
# ==========================================================================

def check_cooperative_names(
    df,
    drop_blank=False,
    drop_inconsistent=False
):

    print("\n")
    print("=" * 80)
    print("COOPERATIVE NAME CHECK")
    print("=" * 80)

    # ------------------------------------------------------
    # Blank / NULL Checks
    # ------------------------------------------------------

    for column in [

        "cooperative_name_en",
        "cooperative_name_np"

    ]:

        if column not in df.columns:

            print(f"Skipping {column} (column missing)\n")
            continue

        blank = (

            df[column].isna()

            |

            (df[column].astype(str).str.strip() == "")

        )

        print(f"{column}")
        print(f"Blank / NULL Rows : {blank.sum():,}")

        if blank.sum():

            print()

            print(
                df.loc[
                    blank,
                    [
                        "cooperative_id",
                        column
                    ]
                ].head(MAX_PRINT_ROWS)
            )

            print()

        if drop_blank:

            before = len(df)

            df = df.loc[
                ~blank
            ].copy()

            print(
                f"Dropped {before - len(df):,} rows "
                f"with blank {column}."
            )

            print()

    # ------------------------------------------------------
    # One cooperative_id -> One English Name
    # ------------------------------------------------------

    if (

        "cooperative_id" in df.columns

        and

        "cooperative_name_en" in df.columns

    ):

        print("=" * 80)
        print("COOPERATIVE ID -> NAME CONSISTENCY")
        print("=" * 80)

        name_counts = (

            df.groupby("cooperative_id")[
                "cooperative_name_en"
            ]
            .nunique()

        )

        inconsistent = name_counts[
            name_counts > 1
        ]

        print(
            f"Inconsistent cooperative IDs : "
            f"{len(inconsistent):,}"
        )

        if len(inconsistent):

            print("\nExample IDs:\n")

            print(
                inconsistent.head(MAX_PRINT_ROWS)
            )

            print("\nExample Records:\n")

            example_ids = (
                inconsistent
                .head(MAX_PRINT_ROWS)
                .index
            )

            print(

                df.loc[
                    df["cooperative_id"].isin(example_ids),
                    [
                        "cooperative_id",
                        "cooperative_name_en",
                        "cooperative_name_np"
                    ]
                ]
                .sort_values(
                    "cooperative_id"
                )

            )

            if drop_inconsistent:

                before = len(df)

                df = df.loc[
                    ~df["cooperative_id"].isin(
                        inconsistent.index
                    )
                ].copy()

                print()

                print(
                    f"Dropped {before - len(df):,} rows "
                    "belonging to inconsistent "
                    "cooperative IDs."
                )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check cooperative_name_en and
    # cooperative_name_np against
    # cooperative_members.csv.
    #
    # Verify every cooperative_id maps to
    # exactly one English name and one
    # Nepali name across all datasets.
    #

    return df

# ==========================================================================
# Cooperative Type Check
# ==========================================================================

def check_cooperative_type(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("COOPERATIVE TYPE CHECK")
    print("=" * 80)

    if require_columns(df, ["cooperative_type"], "COOPERATIVE TYPE CHECK"):
        return df

    distribution = (
        df["cooperative_type"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Cooperative Type Distribution:\n")
    print(distribution)

    invalid = ~df["cooperative_type"].isin(EXPECTED_COOPERATIVE_TYPES)

    print(f"\nInvalid cooperative_type values : {invalid.sum():,}")

    if "cooperative_id" in df.columns:

        print("\nChecking one cooperative_id maps to exactly one "
              "cooperative_type...")

        type_counts = (
            df.groupby("cooperative_id")["cooperative_type"]
            .nunique()
        )

        inconsistent = type_counts[type_counts > 1]

        print(f"cooperative_id values with multiple types : "
              f"{len(inconsistent):,}")

    if drop_invalid:

        df = df.loc[~invalid].copy()

        print("\nInvalid rows removed.")

    return df


# ==========================================================================
# Address Field Check
# ==========================================================================

def check_address_fields(df):

    print("\n")
    print("=" * 80)
    print("ADDRESS FIELD CHECK")
    print("=" * 80)

    if "province_en" in df.columns:

        invalid_province = ~df["province_en"].isin(EXPECTED_PROVINCES)

        print(f"Invalid province_en values : {invalid_province.sum():,}")

    else:

        print("SKIPPING province_en check — column missing.")

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

    print("\nCROSS-TABLE CHECK NEEDED: confirm these address fields match")
    print("the member's applicant_profiles.csv address.")


# ==========================================================================
# Membership Year Check
# ==========================================================================

def check_membership_year(df):

    print("\n")
    print("=" * 80)
    print("MEMBERSHIP YEAR CHECK")
    print("=" * 80)

    if require_columns(df, ["membership_year_bs"], "MEMBERSHIP YEAR CHECK"):
        return

    year = pd.to_numeric(df["membership_year_bs"], errors="coerce")

    out_of_range = (
        year.notna()
        &
        ((year < 2072) | (year > 2080))
    )

    print(f"Out-of-range values (expected 2072–2080) : {out_of_range.sum():,}")

    if out_of_range.sum():

        print("\nExample rows:\n")

        print(
            df.loc[
                out_of_range,
                ["member_id", "membership_year_bs"]
            ].head(MAX_PRINT_ROWS)
        )

    print("\nMembership Year Distribution\n")

    print(year.value_counts(dropna=False).sort_index())


# ==========================================================================
# Share / Dividend Check
# ==========================================================================

def check_shares_and_dividend(
    df,
    drop_inconsistent_totals=False
):

    print("\n")
    print("=" * 80)
    print("SHARE / DIVIDEND CHECK")
    print("=" * 80)

    if "share_count" in df.columns:

        share_count = pd.to_numeric(df["share_count"], errors="coerce")

        out_of_range = (
            share_count.notna()
            &
            ((share_count < 5) | (share_count > 50))
        )

        print(f"share_count out-of-range (expected 5–50) : "
              f"{out_of_range.sum():,}")

    else:

        print("SKIPPING share_count check — column missing.")

    if "share_value_each_nrs" in df.columns:

        share_value = pd.to_numeric(df["share_value_each_nrs"], errors="coerce")

        not_fixed = share_value != 1000

        print(f"\nshare_value_each_nrs != 1000 : {not_fixed.sum():,}")

    else:

        print("\nSKIPPING share_value_each_nrs check — column missing.")

    total_required = [
        "share_count", "share_value_each_nrs", "total_share_value_nrs"
    ]

    if not require_columns(df, total_required, "total_share_value_nrs consistency check"):

        share_count = pd.to_numeric(df["share_count"], errors="coerce")
        share_value = pd.to_numeric(df["share_value_each_nrs"], errors="coerce")
        total = pd.to_numeric(df["total_share_value_nrs"], errors="coerce")

        expected_total = share_count * share_value

        mismatch = (
            total.notna()
            & expected_total.notna()
            & (total != expected_total)
        )

        print(f"\ntotal_share_value_nrs != share_count × "
              f"share_value_each_nrs : {mismatch.sum():,}")

        if mismatch.sum():

            print("\nExample rows:\n")

            print(
                df.loc[
                    mismatch,
                    [
                        "member_id",
                        "share_count",
                        "share_value_each_nrs",
                        "total_share_value_nrs"
                    ]
                ].head(MAX_PRINT_ROWS)
            )

        out_of_range = (
            total.notna()
            &
            ((total < 5000) | (total > 50000))
        )

        print(f"\ntotal_share_value_nrs out-of-range (expected 5,000–")
        print(f"50,000) : {out_of_range.sum():,}")

        if drop_inconsistent_totals:

            df = df.loc[~mismatch].copy()

            print("\nInconsistent rows removed.")

    if "last_annual_dividend_nrs" in df.columns:

        dividend = pd.to_numeric(df["last_annual_dividend_nrs"], errors="coerce")

        non_positive = dividend <= 0

        print(f"\nNon-positive last_annual_dividend_nrs (expected positive) "
              f": {non_positive.sum():,}")

        print("\nDividend Statistics\n")
        print(dividend.describe())

    else:

        print("\nSKIPPING last_annual_dividend_nrs check — column missing.")

    return df


# ==========================================================================
# Outstanding Loan / Repayment Status Check
# ==========================================================================

def check_outstanding_loan(df):

    print("\n")
    print("=" * 80)
    print("OUTSTANDING LOAN / REPAYMENT STATUS CHECK")
    print("=" * 80)

    if "outstanding_loan_nrs" in df.columns:

        loan = pd.to_numeric(df["outstanding_loan_nrs"], errors="coerce")

        negative = loan < 0

        print(f"Negative outstanding_loan_nrs : {negative.sum():,}")

        print("\nOutstanding Loan Statistics\n")
        print(loan.describe())

    else:

        print("SKIPPING outstanding_loan_nrs check — column missing.")

    if "coop_loan_repayment_status" in df.columns:

        distribution = (
            df["coop_loan_repayment_status"]
            .value_counts(dropna=False)
            .to_frame("Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] / len(df) * 100
        ).round(2)

        print("\ncoop_loan_repayment_status Distribution:\n")
        print(distribution)

        invalid = ~df["coop_loan_repayment_status"].isin(
            EXPECTED_REPAYMENT_STATUSES
        )

        print(f"\nInvalid values : {invalid.sum():,}")

        if "outstanding_loan_nrs" in df.columns:

            print("\nChecking coop_loan_repayment_status = 'none' implies "
                  "outstanding_loan_nrs = 0...")

            loan = pd.to_numeric(df["outstanding_loan_nrs"], errors="coerce")

            none_with_balance = (
                (df["coop_loan_repayment_status"] == "none")
                &
                (loan > 0)
            )

            print(f"'none' status but non-zero balance : "
                  f"{none_with_balance.sum():,}")

            current_or_overdue_zero = (
                df["coop_loan_repayment_status"].isin(["current", "overdue"])
                &
                (loan == 0)
            )

            print(f"'current'/'overdue' status but zero balance : "
                  f"{current_or_overdue_zero.sum():,}")

    else:

        print("\nSKIPPING coop_loan_repayment_status check — column missing.")


# ==========================================================================
# Membership Status Check
# ==========================================================================

def check_membership_status(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("MEMBERSHIP STATUS CHECK")
    print("=" * 80)

    if require_columns(df, ["membership_status"], "MEMBERSHIP STATUS CHECK"):
        return df

    distribution = (
        df["membership_status"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print("Membership Status Distribution:\n")
    print(distribution)

    invalid = ~df["membership_status"].isin(EXPECTED_MEMBERSHIP_STATUSES)

    print(f"\nInvalid membership_status values : {invalid.sum():,}")

    if drop_invalid:

        df = df.loc[~invalid].copy()

        print("\nInvalid rows removed.")

    return df


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
        drop_mismatched_applicant=False,
        drop_duplicate_ids=False
    )

    check_cooperative_id(df)

    check_cooperative_names(df,drop_inconsistent=True)

    df = check_cooperative_type(
        df,
        drop_invalid=False
    )

    check_address_fields(df)

    check_membership_year(df)

    df = check_shares_and_dividend(
        df,
        drop_inconsistent_totals=False
    )

    check_outstanding_loan(df)

    df = check_membership_status(
        df,
        drop_invalid=False
    )

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
        r"cooperative_members_ver(\d+)\.csv"
    )

    versions = []

    for file in output_folder.glob("cooperative_members_ver*.csv"):

        match = pattern.fullmatch(file.name)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1

    output_file = (
        output_folder
        / f"cooperative_members_ver{next_version}.csv"
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
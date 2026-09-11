"""
===========================================================================
Cooperative Sales Validator
===========================================================================

Checks

    Dataset Information
    Null Values
    Duplicate Rows
    Sale ID
    Applicant ID
    Cooperative ID
    Cooperative Type
    Province / District / Municipality
    Sale Year
    Season
    Commodity
    Unit
    Quantity
    Rate
    Total Amount
    Amount Consistency

Cross-table validations are marked as TODO.

===========================================================================
"""

import pandas as pd
import re
from pathlib import Path


# ==========================================================================
# Configuration
# ==========================================================================

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "cooperative_sales.csv"
)

MAX_PRINT_ROWS = 10


# ==========================================================================
# Dataset Loading
# ==========================================================================

def load_dataset():

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    print(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    print()

    print("Dataset Loaded Successfully")

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

    print("\nColumns\n")

    for column in df.columns:
        print(column)


# ==========================================================================
# Null Values
# ==========================================================================

def check_nulls(
    df,
    drop_null_rows=False
):

    print("\n")
    print("=" * 80)
    print("NULL VALUE CHECK")
    print("=" * 80)

    summary = pd.DataFrame()

    summary["Null Count"] = df.isnull().sum()

    summary["Percentage"] = (
        summary["Null Count"]
        / len(df)
        * 100
    ).round(2)

    print(summary)

    if drop_null_rows:

        before = len(df)

        df = df.dropna()

        print(f"\nDropped {before-len(df):,} rows.")

    return df


# ==========================================================================
# Duplicate Rows
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

    print(f"Duplicate Rows : {duplicate_rows.sum():,}")

    if duplicate_rows.sum():

        print()

        print(
            df.loc[
                duplicate_rows
            ].head(MAX_PRINT_ROWS)
        )

    if drop_duplicates:

        df = df.drop_duplicates()

        print("\nDuplicate rows removed.")

    return df


# ==========================================================================
# Sale ID
# ==========================================================================

def check_sale_id(
    df,
    drop_invalid=False,
    drop_duplicates=False
):

    print("\n")
    print("=" * 80)
    print("SALE ID CHECK")
    print("=" * 80)

    regex = r"^CSALE-AP-\d{6}-\d{2}$"

    invalid = (
        ~df["sale_id"]
        .astype(str)
        .str.match(regex)
    )

    print(f"Invalid Format : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                ["sale_id"]
            ].head(MAX_PRINT_ROWS)
        )

    duplicate = df["sale_id"].duplicated()

    print()

    print(f"Duplicate IDs : {duplicate.sum():,}")

    if duplicate.sum():

        print()

        print(
            df.loc[
                duplicate,
                ["sale_id"]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    if drop_duplicates:

        df = df.loc[
            ~duplicate
        ].copy()

    return df


# ==========================================================================
# Applicant ID
# ==========================================================================

def check_applicant_id(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("APPLICANT ID CHECK")
    print("=" * 80)

    regex = r"^AP-\d{6}$"

    invalid = (
        ~df["applicant_id"]
        .astype(str)
        .str.match(regex)
    )

    print(f"Invalid Applicant IDs : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "applicant_id"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check applicant_id exists
    # in applicant_profiles.csv
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Cooperative ID
# ==========================================================================

def check_cooperative_id(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("COOPERATIVE ID CHECK")
    print("=" * 80)

    regex = r"^COOP-[A-Z]{3}-\d{4}$"

    invalid = (
        ~df["cooperative_id"]
        .astype(str)
        .str.match(regex)
    )

    print(f"Invalid Cooperative IDs : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "cooperative_id"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Verify cooperative_id exists
    # in cooperative_members.csv
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Cooperative Type
# ==========================================================================

def check_cooperative_type(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("COOPERATIVE TYPE CHECK")
    print("=" * 80)

    expected = {

        "agricultural",
        "dairy",
        "vegetable",
        "coffee_tea",
        "savings_credit"

    }

    distribution = (
        df["cooperative_type"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        / len(df)
        * 100
    ).round(2)

    print(distribution)

    invalid = (
        ~df["cooperative_type"].isin(expected)
    )

    print(f"\nInvalid Values : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "cooperative_type"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check cooperative_type
    # with cooperative_members.csv.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Location Check
# ==========================================================================

def check_location(df):

    print("\n")
    print("=" * 80)
    print("LOCATION CHECK")
    print("=" * 80)

    for column in [

        "province_en",
        "district_en",
        "municipality_en"

    ]:

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

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Province
    #      ↓
    # District
    #      ↓
    # Municipality
    #
    # Validate using Nepal administrative
    # boundary data.
    #
    # Cross-check location with
    # cooperative_members.csv.
    #


# ==========================================================================
# Sale Year
# ==========================================================================

def check_sale_year(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("SALE YEAR CHECK")
    print("=" * 80)

    invalid = (

        (df["sale_year_bs"] < 2078)

        |

        (df["sale_year_bs"] > 2081)

    )

    print(f"Invalid Years : {invalid.sum():,}")
    print(f"Percentage    : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "sale_year_bs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Season
# ==========================================================================

def check_season(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("SEASON CHECK")
    print("=" * 80)

    expected = {

        "Kharif",
        "Rabi",
        "Annual"

    }

    distribution = (
        df["season"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        / len(df)
        * 100
    ).round(2)

    print(distribution)

    invalid = (
        ~df["season"].isin(expected)
    )

    print(f"\nInvalid Values : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "season"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Commodity
# ==========================================================================

def check_commodity(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("COMMODITY CHECK")
    print("=" * 80)

    expected = {

        "Paddy",
        "Vegetables",
        "Maize",
        "Wheat",

        "Milk",
        "Ghee",
        "Curd",

        "Tomato",
        "Potato",
        "Cabbage",
        "Onion",

        "Coffee Beans",
        "Tea Leaves"

    }

    distribution = (
        df["commodity_en"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        / len(df)
        * 100
    ).round(2)

    print(distribution)

    invalid = (
        ~df["commodity_en"].isin(expected)
    )

    print(f"\nInvalid Commodities : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "commodity_en"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Commodity should match
    # cooperative_type.
    #
    # agricultural ->
    #     Paddy
    #     Vegetables
    #     Wheat
    #     Maize
    #
    # dairy ->
    #     Milk
    #     Ghee
    #     Curd
    #
    # vegetable ->
    #     Tomato
    #     Potato
    #     Cabbage
    #     Onion
    #
    # coffee_tea ->
    #     Coffee Beans
    #     Tea Leaves
    #
    # savings_credit ->
    #     should have NO sale records.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Unit
# ==========================================================================

def check_unit(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("UNIT CHECK")
    print("=" * 80)

    expected = {

        "kg",
        "L"

    }

    distribution = (
        df["unit"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        / len(df)
        * 100
    ).round(2)

    print(distribution)

    invalid = (
        ~df["unit"].isin(expected)
    )

    print(f"\nInvalid Units : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "unit"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Commodity determines unit.
    #
    # Milk -> L
    #
    # Everything else -> kg
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Quantity
# ==========================================================================

def check_quantity(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("QUANTITY CHECK")
    print("=" * 80)

    invalid = (
        (df["quantity"] < 50)
        |
        (df["quantity"] > 1500)
    )

    print(f"Invalid Quantities : {invalid.sum():,}")
    print(f"Percentage         : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "quantity"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Rate Per Unit
# ==========================================================================

def check_rate(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("RATE PER UNIT CHECK")
    print("=" * 80)

    invalid = (
        df["rate_nrs_per_unit"] <= 0
    )

    print(f"Invalid Rates : {invalid.sum():,}")
    print(f"Percentage    : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "rate_nrs_per_unit"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Compare with expected market rates.
    #
    # Example:
    #
    # Paddy ≈ 35
    # Milk ≈ 90
    # Coffee Beans ≈ 400
    #
    # Large deviations should be flagged.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Total Amount
# ==========================================================================

def check_total_amount(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("TOTAL AMOUNT CHECK")
    print("=" * 80)

    invalid = (
        df["total_amount_nrs"] <= 0
    )

    print(f"Invalid Amounts : {invalid.sum():,}")
    print(f"Percentage      : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "sale_id",
                    "total_amount_nrs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Amount Consistency
# ==========================================================================

def check_amount_consistency(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("AMOUNT CONSISTENCY CHECK")
    print("=" * 80)

    expected_total = (
        df["quantity"]
        *
        df["rate_nrs_per_unit"]
    )

    invalid = (
        expected_total
        !=
        df["total_amount_nrs"]
    )

    print(f"Inconsistent Records : {invalid.sum():,}")
    print(f"Percentage           : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        example = df.loc[
            invalid,
            [
                "sale_id",
                "quantity",
                "rate_nrs_per_unit",
                "total_amount_nrs"
            ]
        ].copy()

        example["Expected Total"] = (
            example["quantity"]
            *
            example["rate_nrs_per_unit"]
        )

        print(example.head(MAX_PRINT_ROWS))

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Run All Checks
# ==========================================================================

def run_all_checks():

    df = load_dataset()

    dataset_information(df)

    df = check_nulls(df)
    df = check_duplicates(df)

    df = check_sale_id(df)
    df = check_applicant_id(df)
    df = check_cooperative_id(df)

    df = check_cooperative_type(df)

    check_location(df)

    df = check_sale_year(df)
    df = check_season(df)
    df = check_commodity(df)
    df = check_unit(df)

    df = check_quantity(df)
    df = check_rate(df)
    df = check_total_amount(df)

    df = check_amount_consistency(df)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

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
        r"cooperative_sales_ver(\d+)\.csv"
    )

    versions = []

    for file in output_folder.glob("cooperative_sales*.csv"):

        match = pattern.fullmatch(file.name)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1

    output_file = (
        output_folder
        / f"cooperative_sales_ver{next_version}.csv"
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
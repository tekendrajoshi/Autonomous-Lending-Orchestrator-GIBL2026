"""
===========================================================================
Applicant Profiles Validator
===========================================================================

Checks

    Dataset Information
    Null Values
    Duplicate Rows
    Applicant ID
    Names
    Citizenship
    Date of Birth
    Gender
    Marital Status
    Address
    Ward Number
    Phone Number
    Occupation
    Education
    Household Size
    Land Area
    Wallet Information
    Cooperative Information
    KYC
    Rural / Urban
    Profile Created Date

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
    / "applicant_profiles.csv"
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

        print()

        print(f"Dropped {before-len(df):,} rows.")

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

    duplicates = df.duplicated()

    print(f"Duplicate Rows : {duplicates.sum():,}")

    if duplicates.sum():

        print()

        print(
            df.loc[
                duplicates
            ].head(MAX_PRINT_ROWS)
        )

    if drop_duplicates:

        df = df.drop_duplicates()

        print()

        print("Duplicate rows removed.")

    return df


# ==========================================================================
# Applicant ID
# ==========================================================================

def check_applicant_id(
    df,
    drop_invalid=False,
    drop_duplicates=False
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

    print(f"Invalid Format : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(

            df.loc[
                invalid,
                ["applicant_id"]
            ].head(MAX_PRINT_ROWS)

        )

    duplicate = df["applicant_id"].duplicated()

    print()

    print(f"Duplicate IDs : {duplicate.sum():,}")

    if duplicate.sum():

        print()

        print(

            df.loc[
                duplicate,
                ["applicant_id"]
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
# Name Checks
# ==========================================================================

# ==========================================================================
# Name Checks
# ==========================================================================

# ==========================================================================
# Name Checks
# ==========================================================================

def check_names(
    df,
    all_lower=False
):

    print("\n")
    print("=" * 80)
    print("NAME CHECKS")
    print("=" * 80)

    name_columns = [

        "full_name_en",
        "father_name_en",
        "grandfather_name_en"

    ]

    for column in name_columns:

        if column not in df.columns:

            continue

        print()

        print(column)

        upper = (
            df[column]
            ==
            df[column].str.upper()
        )

        lower = (
            df[column]
            ==
            df[column].str.lower()
        )

        print(f"ALL CAPS  : {upper.sum():,}")
        print(f"all lower : {lower.sum():,}")

        if upper.sum():

            print("\nExample ALL CAPS values:\n")

            print(
                df.loc[
                    upper,
                    [column]
                ].head(MAX_PRINT_ROWS)
            )

        if lower.sum():

            print("\nExample all lower values:\n")

            print(
                df.loc[
                    lower,
                    [column]
                ].head(MAX_PRINT_ROWS)
            )

        if all_lower:

            print(f"\nConverting {column} to lowercase...")

            df[column] = (

                df[column]

                .fillna("")

                .astype(str)

                .str.lower()

                .str.strip()

            )

            print("Done.")

    # ------------------------------------------------------
    # TODO
    #
    # Cross-check names against OCR extracted
    # document names.
    #
    # Check for duplicated names sharing
    # different applicant IDs.
    #
    # Compare English names against
    # Nepali transliterations if available.
    #

    return df


# ==========================================================================
# Citizenship Number
# ==========================================================================

def check_citizenship_number(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("CITIZENSHIP NUMBER CHECK")
    print("=" * 80)

    regex = r"^\d{3}-0\d{2}-\d{5}$"

    invalid = (
        ~df["citizenship_number"]
        .astype(str)
        .str.match(regex)
    )

    print(f"Invalid Citizenship Numbers : {invalid.sum():,}")
    print(f"Percentage                  : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print("\nExample Invalid Rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "citizenship_number"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    # Cross-check district code (first 3 digits)
    # with district_en once district code mapping
    # is available.

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Citizenship Date (BS)
# ==========================================================================

def check_citizenship_date_bs(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("CITIZENSHIP DATE (BS) CHECK")
    print("=" * 80)

    regex = r"^\d{4}/\d{2}/\d{2}$"

    invalid = (
        ~df["citizenship_date_bs"]
        .astype(str)
        .str.match(regex)
    )

    print(f"Invalid Format : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "citizenship_date_bs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    # Convert BS date to AD.
    #
    # Check:
    # citizenship_date > DOB
    #
    # citizenship_date < profile_created_date

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Date of Birth (AD)
# ==========================================================================

def check_dob_ad(
    df,
    earliest_valid="1924-01-01",
    latest_valid=None,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("DATE OF BIRTH (AD) CHECK")
    print("=" * 80)

    if latest_valid is None:

        latest_valid = pd.Timestamp.now()

    else:

        latest_valid = pd.Timestamp(latest_valid)

    earliest_valid = pd.Timestamp(earliest_valid)

    parsed = pd.to_datetime(
        df["dob_ad"],
        errors="coerce"
    )

    invalid = (

        parsed.isna()

        |

        (parsed < earliest_valid)

        |

        (parsed > latest_valid)

    )

    print(f"Impossible Dates : {invalid.sum():,}")
    print(f"Percentage       : {invalid.mean()*100:.2f}%")

    print()

    print(f"Earliest DOB : {parsed.min()}")
    print(f"Latest DOB   : {parsed.max()}")

    if invalid.sum():

        print("\nExample Invalid Rows:\n")

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "dob_ad"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    # Calculate age.
    #
    # Check age is between
    # 18 and 70 years.
    #
    # Compare with dob_bs.

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Date of Birth (BS)
# ==========================================================================

def check_dob_bs(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("DATE OF BIRTH (BS) CHECK")
    print("=" * 80)

    regex = r"^\d{4}-\d{2}-\d{2}$"

    invalid = (
        ~df["dob_bs"]
        .astype(str)
        .str.match(regex)
    )

    print(f"Invalid Format : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "dob_bs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    # Verify BS date corresponds to AD DOB.

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Gender
# ==========================================================================

def check_gender(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("GENDER CHECK")
    print("=" * 80)

    expected = {
        "male",
        "female"
    }

    distribution = (
        df["gender"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        / len(df)
        * 100
    ).round(2)

    print(distribution)

    invalid = ~df["gender"].isin(expected)

    print(f"\nInvalid Values : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "gender"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Marital Status
# ==========================================================================

def check_marital_status(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("MARITAL STATUS CHECK")
    print("=" * 80)

    expected = {

        "Married",
        "Single",
        "Widowed",
        "Divorced"

    }

    distribution = (
        df["marital_status"]
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
        ~df["marital_status"]
        .isin(expected)
    )

    print(f"\nInvalid Values : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "marital_status"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df

# ==========================================================================
# Address Check
# ==========================================================================

def check_address(df):

    print("\n")
    print("=" * 80)
    print("ADDRESS CHECK")
    print("=" * 80)

    print("\nProvince Distribution\n")

    province = (
        df["province_en"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    province["Percentage"] = (
        province["Count"] / len(df) * 100
    ).round(2)

    print(province)

    print("\nDistrict Distribution\n")

    district = (
        df["district_en"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    district["Percentage"] = (
        district["Count"] / len(df) * 100
    ).round(2)

    print(district)

    print("\nMunicipality Distribution\n")

    municipality = (
        df["municipality_en"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    municipality["Percentage"] = (
        municipality["Count"] / len(df) * 100
    ).round(2)

    print(municipality)

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check:
    #
    # Province
    #      ↓
    # District
    #      ↓
    # Municipality
    #
    # using Nepal administrative boundary data.
    #
    # Dataset documentation mentions approximately
    # 2.5% intentionally incorrect combinations.
    #
    # Example:
    #
    # Province = Bagmati
    # District = Jhapa
    #
    # should be detected.
    #


# ==========================================================================
# Ward Number
# ==========================================================================

def check_ward_number(
    df,
    drop_null=False,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("WARD NUMBER CHECK")
    print("=" * 80)

    null_rows = df["ward_no"].isna()

    print(f"NULL Values : {null_rows.sum():,}")
    print(f"Percentage  : {null_rows.mean()*100:.2f}%")

    invalid = (

        df["ward_no"].notna()

        &

        (
            (df["ward_no"] <= 0)

            |

            (df["ward_no"] > 35)
        )

    )

    print()

    print(f"Invalid Ward Numbers : {invalid.sum():,}")
    print(f"Percentage           : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "municipality_en",
                    "ward_no"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Check maximum ward number for
    # each municipality.
    #
    # Example:
    #
    # Banepa Municipality
    # max ward = 14
    #
    # Ward 23 should be invalid.
    #
    if invalid.sum():

        print()
        print("=" * 80)
        print("INVESTIGATING INVALID WARD ROWS")
        print("=" * 80)

        invalid_rows = df.loc[invalid]

        print(f"Rows investigated : {len(invalid_rows):,}")

        print("\nMissing values in other columns:\n")

        print(
            invalid_rows.isna().sum()
        )

        print("\nProvince Distribution:\n")

        print(
            invalid_rows["province_en"]
            .value_counts(dropna=False)
        )

        print("\nDistrict Distribution:\n")

        print(
            invalid_rows["district_en"]
            .value_counts(dropna=False)
        )

        print("\nMunicipality Distribution:\n")

        print(
            invalid_rows["municipality_en"]
            .value_counts(dropna=False)
            .head(20)
        )

        print(
            "\nNOTE:"
            "\nIf province, district, municipality and applicant_id "
            "appear reasonable, the injected noise likely affects "
            "only ward_no. Since ward_no is not used as a feature, "
            "these rows can usually be retained."
        )



    if drop_null:

        df = df.loc[
            ~null_rows
        ].copy()

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Primary Phone
# ==========================================================================

def check_phone_primary(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("PRIMARY PHONE CHECK")
    print("=" * 80)

    null_rows = df["phone_primary"].isna()

    print(f"NULL Phones : {null_rows.sum():,}")
    print(f"Percentage  : {null_rows.mean()*100:.2f}%")

    phone = (
        df["phone_primary"]
        .fillna("")
        .astype(str)
    )

    invalid = (

        (~phone.str.fullmatch(r"\d{10}"))

        &

        (~null_rows)

    )

    print()

    print(f"Invalid Phone Numbers : {invalid.sum():,}")
    print(f"Percentage            : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "phone_primary"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # Investigation:
    # Are invalid phone rows otherwise valid?
    # ------------------------------------------------------

    if invalid.sum():

        print()
        print("=" * 80)
        print("INVESTIGATING INVALID PHONE ROWS")
        print("=" * 80)

        invalid_rows = df.loc[invalid]

        print(f"Rows investigated : {len(invalid_rows):,}")

        print("\nPhone number lengths:\n")

        print(
            invalid_rows["phone_primary"]
            .astype(str)
            .str.len()
            .value_counts()
            .sort_index()
        )

        print("\nMissing values in other columns:\n")

        print(
            invalid_rows.isna().sum()
        )

        print("\nOccupation distribution:\n")

        print(
            invalid_rows["occupation_en"]
            .value_counts(dropna=False)
        )

        print("\nProvince distribution:\n")

        print(
            invalid_rows["province_en"]
            .value_counts(dropna=False)
        )

        print("\nKYC Tier distribution:\n")

        print(
            invalid_rows["kyc_tier"]
            .value_counts(dropna=False)
        )

        print(
            "\nNOTE:"
            "\nIf these rows otherwise appear normal, "
            "the injected noise likely affects only "
            "phone_primary. Since phone_primary is not "
            "used as a model feature, these rows can "
            "usually be retained."
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Dataset states generated numbers
    # deliberately DO NOT begin with
    # 97, 98 or 99.
    #
    # Check this later.
    #
    # Cross-check phone numbers against
    # other tables if they ever appear.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df 


# ==========================================================================
# Occupation
# ==========================================================================

def check_occupation(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("OCCUPATION CHECK")
    print("=" * 80)

    expected = {

        "Farmer",
        "Daily Wage Worker",
        "Small Trader",
        "Service Worker",
        "Remittance Dependent",
        "Artisan",
        "Government Employee",
        "Business Owner",
        "Teacher",
        "Driver",
        "Nurse/Health Worker"

    }

    distribution = (
        df["occupation_en"]
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
        ~df["occupation_en"]
        .isin(expected)
    )

    print(f"\nInvalid Occupations : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "occupation_en"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Compare occupation_en with
    # occupation_np translation.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Education Level
# ==========================================================================

def check_education_level(
    df,
    drop_null=False,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("EDUCATION LEVEL CHECK")
    print("=" * 80)

    expected = {

        "Illiterate",
        "Primary",
        "SLC",
        "Intermediate",
        "Bachelors",
        "Masters",
        "PhD"

    }

    distribution = (
        df["education_level"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        / len(df)
        * 100
    ).round(2)

    print(distribution)

    null_rows = df["education_level"].isna()

    print()

    print(f"NULL Values : {null_rows.sum():,}")
    print(f"Percentage  : {null_rows.mean()*100:.2f}%")

    invalid = (

        df["education_level"].notna()

        &

        ~df["education_level"].isin(expected)

    )

    print()

    print(f"Invalid Values : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "education_level"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_null:

        df = df.loc[
            ~null_rows
        ].copy()

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df

# ==========================================================================
# Household Size
# ==========================================================================

def check_household_size(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("HOUSEHOLD SIZE CHECK")
    print("=" * 80)

    invalid = (
        (df["household_size"] < 1)
        |
        (df["household_size"] > 8)
    )

    print(f"Invalid Household Sizes : {invalid.sum():,}")
    print(f"Percentage              : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "household_size"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Land Area
# ==========================================================================

def check_land_area(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("LAND AREA CHECK")
    print("=" * 80)

    invalid = (
        (df["land_area_ropani"] < 0)
        |
        (df["land_area_ropani"] > 21)
    )

    print(f"Invalid Land Areas : {invalid.sum():,}")
    print(f"Percentage         : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "land_area_ropani"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Wallet Information
# ==========================================================================

def check_wallet_information(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("WALLET INFORMATION CHECK")
    print("=" * 80)

    # ------------------------------------------------------
    # eSewa
    # ------------------------------------------------------

    print("\nChecking eSewa Accounts")

    esewa_invalid = (

        (df["has_esewa_account"])

        &

        (df["esewa_account_id"].isna())

    ) | (

        (~df["has_esewa_account"])

        &

        (df["esewa_account_id"].notna())

    )

    print(f"Inconsistent eSewa Records : {esewa_invalid.sum():,}")

    if esewa_invalid.sum():

        print()

        print(
            df.loc[
                esewa_invalid,
                [
                    "applicant_id",
                    "has_esewa_account",
                    "esewa_account_id"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    regex = r"^ESW-AP-\d{6}$"

    invalid_format = (

        df["esewa_account_id"].notna()

        &

        ~df["esewa_account_id"]
        .str.match(regex)

    )

    print()

    print(f"Invalid eSewa IDs : {invalid_format.sum():,}")

    # ------------------------------------------------------
    # Khalti
    # ------------------------------------------------------

    print("\nChecking Khalti Accounts")

    khalti_invalid = (

        (df["has_khalti_account"])

        &

        (df["khalti_account_id"].isna())

    ) | (

        (~df["has_khalti_account"])

        &

        (df["khalti_account_id"].notna())

    )

    print(f"Inconsistent Khalti Records : {khalti_invalid.sum():,}")

    if khalti_invalid.sum():

        print()

        print(
            df.loc[
                khalti_invalid,
                [
                    "applicant_id",
                    "has_khalti_account",
                    "khalti_account_id"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    regex = r"^KHL-AP-\d{6}$"

    invalid_format = (

        df["khalti_account_id"].notna()

        &

        ~df["khalti_account_id"]
        .str.match(regex)

    )

    print()

    print(f"Invalid Khalti IDs : {invalid_format.sum():,}")

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check account IDs with
    # mobile_money_transactions.
    #
    # Ensure applicants marked as having
    # wallets actually have transactions.
    #

    if drop_invalid:

        pass

    return df


# ==========================================================================
# Primary Bank
# ==========================================================================

def check_primary_bank(df):

    print("\n")
    print("=" * 80)
    print("PRIMARY BANK CHECK")
    print("=" * 80)

    distribution = (
        df["primary_bank"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print(distribution)

    # TODO
    # Validate against list of
    # 12 Nepali commercial banks.


# ==========================================================================
# Cooperative Information
# ==========================================================================

def check_cooperative(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("COOPERATIVE CHECK")
    print("=" * 80)

    invalid = (

        (df["cooperative_member"])

        &

        (df["cooperative_id"].isna())

    ) | (

        (~df["cooperative_member"])

        &

        (df["cooperative_id"].notna())

    )

    print(f"Inconsistent Records : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "applicant_id",
                    "cooperative_member",
                    "cooperative_id"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    regex = r"^COOP-[A-Z]{3}-\d{4}$"

    invalid_format = (

        df["cooperative_id"].notna()

        &

        ~df["cooperative_id"]
        .str.match(regex)

    )

    print()

    print(f"Invalid Cooperative IDs : {invalid_format.sum():,}")

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Cross-check cooperative_id with
    # cooperative_members.csv
    #
    # Detect ghost memberships.
    #

    if drop_invalid:

        pass

    return df


# ==========================================================================
# KYC Tier
# ==========================================================================

def check_kyc(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("KYC CHECK")
    print("=" * 80)

    expected = {
        "basic",
        "mid",
        "full"
    }

    distribution = (
        df["kyc_tier"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print(distribution)

    invalid = ~df["kyc_tier"].isin(expected)

    print(f"\nInvalid Values : {invalid.sum():,}")

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Rural / Urban
# ==========================================================================

def check_rural_urban(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("RURAL / URBAN CHECK")
    print("=" * 80)

    expected = {
        "rural",
        "semi_urban",
        "urban"
    }

    distribution = (
        df["rural_urban"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100
    ).round(2)

    print(distribution)

    invalid = ~df["rural_urban"].isin(expected)

    print(f"\nInvalid Values : {invalid.sum():,}")

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Profile Created Date
# ==========================================================================

def check_profile_created_date(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("PROFILE CREATED DATE CHECK")
    print("=" * 80)

    parsed = pd.to_datetime(
        df["profile_created_date"],
        errors="coerce"
    )

    invalid = parsed.isna()

    print(f"Invalid Dates : {invalid.sum():,}")

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Check profile creation date
    # occurs after DOB.
    #
    # Check profile creation date
    # occurs after citizenship issue date.
    #

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

    df = check_nulls(df, False)
    df = check_duplicates(df, False)

    df = check_applicant_id(df, drop_duplicates=True)

    df = check_names(df,all_lower=True)

    df = check_citizenship_number(df, drop_invalid=True)
    df = check_citizenship_date_bs(df)

    df = check_dob_ad(df)
    df = check_dob_bs(df)

    df = check_gender(df)
    df = check_marital_status(df)

    check_address(df)

    df = check_ward_number(df)
    df = check_phone_primary(df)

    df = check_occupation(df)
    df = check_education_level(df)

    df = check_household_size(df)
    df = check_land_area(df)

    df = check_wallet_information(df)

    check_primary_bank(df)

    df = check_cooperative(df)

    df = check_kyc(df)
    df = check_rural_urban(df)

    df = check_profile_created_date(df)

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
        r"applicant_profiles_ver(\d+)\.csv"
    )

    versions = []

    for file in output_folder.glob("applicant_profiles*.csv"):

        match = pattern.fullmatch(file.name)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1

    output_file = (
        output_folder
        / f"applicant_profiles_ver{next_version}.csv"
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
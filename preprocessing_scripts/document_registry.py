"""
===========================================================================
Document Registry Validator
===========================================================================

Checks

    Dataset Information
    Null Values
    Duplicate Rows
    Document ID
    Applicant ID
    Document Type
    Document Subtype
    File Path
    File Format
    Page Count
    Scan DPI
    OCR Complexity
    Languages
    Rotation
    Upload Date
    OCR Baseline CER
    Verification
    Verification Confidence
    Anomaly Flag
    Ground Truth Path

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
    / "document_registry.csv"
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
# Document ID
# ==========================================================================

def check_document_id(
    df,
    drop_invalid=False,
    drop_duplicates=False
):

    print("\n")
    print("=" * 80)
    print("DOCUMENT ID CHECK")
    print("=" * 80)

    regex = r"^DOC-[A-Z]{3}-\d{7}$"

    invalid = (
        ~df["document_id"]
        .astype(str)
        .str.match(regex)
    )

    print(f"Invalid IDs : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                ["document_id"]
            ].head(MAX_PRINT_ROWS)
        )

    duplicate = df["document_id"].duplicated()

    print()

    print(f"Duplicate IDs : {duplicate.sum():,}")

    if duplicate.sum():

        print()

        print(
            df.loc[
                duplicate,
                ["document_id"]
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
                    "document_id",
                    "applicant_id"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    # ------------------------------------------------------
    #
    # Verify applicant_id exists
    # in applicant_profiles.csv.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Document Type
# ==========================================================================

def check_document_type(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("DOCUMENT TYPE CHECK")
    print("=" * 80)

    expected = {

        "citizenship_certificate",
        "utility_bill",
        "kyc_form",
        "lalpurja",
        "cooperative_passbook",
        "remittance_receipt"

    }

    distribution = (
        df["document_type"]
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
        ~df["document_type"].isin(expected)
    )

    print(f"\nInvalid Types : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "document_type"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    #
    # Cross-check applicant profile.
    #
    # Example:
    #
    # land_area_ropani == 0
    # but Lalpurja exists.
    #
    # cooperative_member == False
    # but cooperative_passbook exists.
    #
    # remittance_receiving == False
    # but remittance_receipt exists.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Document Subtype
# ==========================================================================

def check_document_subtype(df):

    print("\n")
    print("=" * 80)
    print("DOCUMENT SUBTYPE")
    print("=" * 80)

    distribution = (
        df["document_subtype"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        / len(df)
        * 100
    ).round(2)

    print(distribution)

    # TODO
    # Validate subtype values once
    # documentation is available.

# ==========================================================================
# File Path
# ==========================================================================

def check_file_path(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("FILE PATH CHECK")
    print("=" * 80)

    expected = (
        "docs/"
        + df["document_type"]
        + "/"
        + df["document_id"]
        + ".jpg"
    )

    invalid = (
        df["file_path"]
        !=
        expected
    )

    print(f"Invalid Paths : {invalid.sum():,}")

    if invalid.sum():

        print()

        example = df.loc[
            invalid,
            [
                "document_id",
                "document_type",
                "file_path"
            ]
        ].copy()

        example["Expected"] = expected[invalid]

        print(
            example.head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# File Format
# ==========================================================================

def check_file_format(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("FILE FORMAT CHECK")
    print("=" * 80)

    distribution = (
        df["file_format"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        /
        len(df)
        *
        100
    ).round(2)

    print(distribution)

    invalid = (
        df["file_format"]
        !=
        "jpg"
    )

    print(f"\nInvalid Formats : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "file_format"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Page Count
# ==========================================================================

def check_page_count(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("PAGE COUNT CHECK")
    print("=" * 80)

    invalid = (
        ~df["page_count"].isin([1, 2])
    )

    print(f"Invalid Page Counts : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "page_count"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    #
    # cooperative_passbook
    # should have 2 pages.
    #
    # Every other document
    # should have exactly
    # 1 page.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Scan DPI
# ==========================================================================

def check_scan_dpi(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("SCAN DPI CHECK")
    print("=" * 80)

    expected = {

        72,
        96,
        150,
        200,
        300

    }

    distribution = (
        df["scan_dpi"]
        .value_counts(dropna=False)
        .sort_index()
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        /
        len(df)
        *
        100
    ).round(2)

    print(distribution)

    invalid = (
        ~df["scan_dpi"].isin(expected)
    )

    print(f"\nInvalid DPI Values : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "scan_dpi"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# OCR Complexity
# ==========================================================================

def check_ocr_complexity(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("OCR COMPLEXITY CHECK")
    print("=" * 80)

    expected = {

        "clean",
        "stamp_overlay",
        "shadow",
        "blur",
        "compression"

    }

    distribution = (
        df["ocr_complexity_tag"]
        .value_counts(dropna=False)
        .to_frame("Count")
    )

    distribution["Percentage"] = (
        distribution["Count"]
        /
        len(df)
        *
        100
    ).round(2)

    print(distribution)

    invalid = (
        ~df["ocr_complexity_tag"]
        .isin(expected)
    )

    print(f"\nInvalid Tags : {invalid.sum():,}")

    if invalid.sum():

        print()

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "ocr_complexity_tag"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Language
# ==========================================================================

def check_languages(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("LANGUAGE CHECK")
    print("=" * 80)

    expected = {

        "nepali",
        "english"

    }

    print("\nPrimary Language\n")

    print(
        df["language_primary"]
        .value_counts(dropna=False)
    )

    primary_invalid = (
        ~df["language_primary"]
        .isin(expected)
    )

    print(f"\nInvalid Primary Languages : {primary_invalid.sum():,}")

    print("\nSecondary Language\n")

    print(
        df["language_secondary"]
        .value_counts(dropna=False)
    )

    secondary_invalid = (

        df["language_secondary"].notna()

        &

        ~df["language_secondary"].isin(expected)

    )

    print(f"\nInvalid Secondary Languages : {secondary_invalid.sum():,}")

    # ------------------------------------------------------
    # TODO
    #
    # Lalpurja documents
    # should have NULL
    # secondary language.
    #
    # Most citizenship
    # certificates should
    # be bilingual.
    #

    if drop_invalid:

        df = df.loc[
            ~(primary_invalid | secondary_invalid)
        ].copy()

    return df

# ==========================================================================
# Rotation
# ==========================================================================

def check_rotation(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("ROTATION CHECK")
    print("=" * 80)

    invalid = (
        (df["rotation_angle_degrees"] < -15)
        |
        (df["rotation_angle_degrees"] > 15)
    )

    print(f"Rotation Outside Range : {invalid.sum():,}")
    print(f"Percentage             : {invalid.mean()*100:.2f}%")

    if invalid.sum():

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "rotation_angle_degrees"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    #
    # If is_rotated == False,
    # rotation_angle_degrees
    # should be 0.
    #
    # If is_rotated == True,
    # angle should normally
    # be non-zero.
    #

    if drop_invalid:

        df = df.loc[
            ~invalid
        ].copy()

    return df


# ==========================================================================
# Upload Date
# ==========================================================================

def check_upload_date(df):

    print("\n")
    print("=" * 80)
    print("UPLOAD DATE CHECK")
    print("=" * 80)

    parsed = pd.to_datetime(
        df["upload_date_bs"],
        errors="coerce"
    )

    invalid = parsed.isna()

    print(f"Invalid Dates : {invalid.sum():,}")

    if invalid.sum():

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "upload_date_bs"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # TODO
    # Validate BS calendar properly.
    # Current check only verifies parseability.


# ==========================================================================
# OCR Baseline CER
# ==========================================================================

def check_ocr_baseline(df):

    print("\n")
    print("=" * 80)
    print("OCR BASELINE CER CHECK")
    print("=" * 80)

    invalid = (
        (df["ocr_model_baseline_cer"] < 0.03)
        |
        (df["ocr_model_baseline_cer"] > 0.21)
    )

    print(f"Outside Expected Range : {invalid.sum():,}")

    if invalid.sum():

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "ocr_model_baseline_cer"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # TODO
    # Compare CER with OCR
    # complexity tag.
    #
    # clean -> ~0.05
    # blur -> ~0.14
    # compression -> ~0.18


# ==========================================================================
# Verification
# ==========================================================================

def check_verification(df):

    print("\n")
    print("=" * 80)
    print("VERIFICATION CHECK")
    print("=" * 80)

    print("\nVerified By Agent")

    print(
        df["verified_by_agent"]
        .value_counts(dropna=False)
    )

    invalid = (
        (df["verification_confidence"] < 0)
        |
        (df["verification_confidence"] > 1)
    )

    print(f"\nInvalid Confidence : {invalid.sum():,}")

    if invalid.sum():

        print(
            df.loc[
                invalid,
                [
                    "document_id",
                    "verification_confidence"
                ]
            ].head(MAX_PRINT_ROWS)
        )

    # ------------------------------------------------------
    # TODO
    #
    # Documents marked
    # verified=False
    # should generally have
    # lower confidence.
    #


# ==========================================================================
# Anomaly Flag
# ==========================================================================

def check_anomaly_flag(df):

    print("\n")
    print("=" * 80)
    print("ANOMALY FLAG")
    print("=" * 80)

    print(
        df["anomaly_flag"]
        .value_counts(dropna=False)
    )


# ==========================================================================
# Ground Truth Path
# ==========================================================================

def check_ground_truth_path(
    df,
    drop_invalid=False
):

    print("\n")
    print("=" * 80)
    print("GROUND TRUTH PATH CHECK")
    print("=" * 80)

    expected = (
        "ground_truth/"
        + df["document_type"]
        + "/"
        + df["document_id"]
        + "_gt.json"
    )

    invalid = (
        df["ground_truth_path"]
        !=
        expected
    )

    print(f"Invalid Paths : {invalid.sum():,}")

    if invalid.sum():

        example = df.loc[
            invalid,
            [
                "document_id",
                "ground_truth_path"
            ]
        ].copy()

        example["Expected"] = expected[invalid]

        print(
            example.head(MAX_PRINT_ROWS)
        )

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

    df = check_document_id(df)
    df = check_applicant_id(df)

    df = check_document_type(df)
    check_document_subtype(df)

    df = check_file_path(df)
    df = check_file_format(df)
    df = check_page_count(df)
    df = check_scan_dpi(df)
    df = check_ocr_complexity(df)

    df = check_languages(df)

    df = check_rotation(df)

    check_upload_date(df)
    check_ocr_baseline(df)
    check_verification(df)
    check_anomaly_flag(df)

    df = check_ground_truth_path(df)

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
        r"document_registry_ver(\d+)\.csv"
    )

    versions = []

    for file in output_folder.glob(
        "document_registry*.csv"
    ):

        match = pattern.fullmatch(file.name)

        if match:

            versions.append(
                int(match.group(1))
            )

    next_version = max(
        versions,
        default=0
    ) + 1

    output_file = (
        output_folder
        / f"document_registry_ver{next_version}.csv"
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
# GIBL AI/ML Hackathon 2026 — Track A
## Autonomous Credit & Lending Orchestrator
# Data Description & Technical Reference Manual

---

> **Version:** 1.0  
> **Classification:** Participant-Accessible  
> **Domain:** Alternative Credit Scoring | Banking & Lending | Document Intelligence

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Entity Relationship & Data Model](#2-entity-relationship--data-model)
3. [Table-by-Table Data Dictionary](#3-table-by-table-data-dictionary)
   - 3.1 applicant_profiles
   - 3.2 loan_applications
   - 3.3 mobile_money_transactions
   - 3.4 remittance_records
   - 3.5 utility_payments
   - 3.6 cooperative_members
   - 3.7 cooperative_sales
   - 3.8 document_registry
4. [Noise Columns Reference](#4-noise-columns-reference)
5. [Banking & Financial Terminology Glossary](#5-banking--financial-terminology-glossary)
6. [Nepal-Specific Terms & Context](#6-nepal-specific-terms--context)
7. [Data Formats & Conventions](#7-data-formats--conventions)
8. [Evaluation Metrics Reference](#8-evaluation-metrics-reference)

---

## 1. Dataset Overview

### 1.1 Purpose

This dataset simulates the operational data environment of a Nepali financial institution
attempting to extend credit to **underserved, credit-invisible populations** — smallholder
farmers, daily wage workers, rural cooperatives, and migrant workers' families — who lack
formal payslips, tax returns, or credit bureau histories.

The dataset supports the following ML/AI tasks:

| Task | Primary Tables Used |
|---|---|
| **Alternative Credit Scoring** | applicant_profiles, loan_applications, mobile_money_transactions, remittance_records, cooperative_members/sales, utility_payments |
| **Document Intelligence / OCR** | document_registry + ground_truth/ folder |
| **Anomaly & Fraud Detection** | mobile_money_transactions (hidden `_noise_anomaly_type` labels) |
| **Income Estimation** | mobile_money_transactions, remittance_records, cooperative_sales |
| **NRB Compliance Checking** | loan_applications (interest_rate_pct, nrb_blacklist_flag, aml_flag, etc.) |
| **Data Cleaning / EDA** | All tables (noise columns guide what to find) |
| **LLM / RAG** | document_registry + ground_truth JSON files |

### 1.2 Scale Summary - Example

| Table | Rows | Format | Size |
|---|---|---|---|
| applicant_profiles | 100,000 | CSV | 41 MB |
| loan_applications | 100,000 | CSV | 28 MB |
| mobile_money_transactions | 2,266,566 | **Parquet** | 49 MB |
| utility_payments | 1,023,855 | **Parquet** | 19 MB |
| remittance_records | 203,312 | **Parquet** | 5.8 MB |
| cooperative_members | 40,757 | CSV | 9.4 MB |
| cooperative_sales | 48,923 | CSV | 6.2 MB |
| document_registry | 414,863 | CSV | 95 MB |

> **Parquet files** require pandas (`pd.read_parquet(...)`) or PySpark. CSV previews of
> each Parquet file are available in the `previews/` folder.

### 1.3 Key Design Decisions

- **Not all customers have all data types.** A customer appears in `cooperative_members`
  only if they are a cooperative member (~42%). They appear in `remittance_records` only if
  they receive international remittances (~36%). This sparsity is **intentional and realistic**.
- **Noise is embedded** in every table. Columns prefixed `_noise_` flag injected anomalies.
  These flags are present in train/val/public_test splits but **removed from the hidden test
  split** — participants must detect anomalies independently.
- **Geography is validated.** All province → district → municipality combinations are real
  administrative units of Nepal, except ~2.5% that are intentionally wrong (geographic noise).

---

## 2. Entity Relationship & Data Model

```
applicant_profiles (1)
    │
    ├──── (1:1) loan_applications          [every applicant has exactly one application]
    │
    ├──── (1:0..N) mobile_money_transactions [only wallet users; ~69% of customers]
    │
    ├──── (1:0..N) remittance_records       [only remittance receivers; ~36%]
    │
    ├──── (1:0..1) cooperative_members      [only coop members; ~41%; ghost noise: 3%]
    │         └── (1:0..N) cooperative_sales [commodity sales through coop]
    │
    ├──── (1:0..N) utility_payments         [NEA electricity + Ncell mobile; ~98%]
    │
    └──── (1:0..6) document_registry        [up to 6 document types per customer]
               └── ground_truth/{type}/{doc_id}_gt.json [OCR annotations]
```

**Primary key:** `applicant_id` (format: `AP-NNNNNN`, e.g. `AP-050234`)

**Foreign keys:**
- All tables reference `applicant_id` from `applicant_profiles`
- `cooperative_sales.cooperative_id` references `cooperative_members.cooperative_id`
- `document_registry.ground_truth_path` points to JSON annotation files

---

## 3. Table-by-Table Data Dictionary

---

### 3.1 `applicant_profiles.csv`

**Description:** Master record for each customer/loan applicant. One row per person.
Contains personal demographics, geographic address, financial profile flags, and
KYC classification. This is the root table — all other tables join to it via `applicant_id`.

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `applicant_id` | STRING | No | `AP-000001` to `AP-100000` | **Primary key.** Unique synthetic identifier for each applicant. Format: `AP-` followed by 6-digit zero-padded integer. Never reused across tables. |
| `full_name_en` | STRING | No | Free text | Applicant's full name in English (First + Surname). May contain noise: ALL CAPS (~3%), all lower (~2%). |
| `father_name_en` | STRING | No | Free text | Father's full name in English. Used in Lalpurja and citizenship document verification. |
| `grandfather_name_en` | STRING | No | Free text | Paternal grandfather's name (for male applicants) or father-in-law's name (for female applicants). Nepal government documents require this field. |
| `citizenship_number` | STRING | No | `DDD-0YY-NNNNN` (e.g. `045-076-12345`) | Synthetic Nepali citizenship certificate number. Format: district-code (3 digits) + hyphen + `0` + year-of-issue (2 digits) + hyphen + serial number (5 digits). **Noise:** ~2.5% are malformed (`CITNOISE` prefix). |
| `citizenship_date_bs` | STRING | No | `YYYY/MM/DD` in BS | Date citizenship was issued, in Bikram Sambat calendar (e.g. `2065/04/15`). |
| `citizenship_office` | STRING | No | Nepali text | Name of the District Administration Office that issued citizenship (e.g. `जिल्ला प्रशासन कार्यालय, Kavrepalanchok`). |
| `dob_ad` | STRING | No | `YYYY-MM-DD` | Date of birth in Anno Domini (Gregorian) calendar. Normal range: 1954–2006 (ages 18–70). **Noise:** ~0.8% future dates (impossible), ~0.5% pre-1924 (too old). |
| `dob_bs` | STRING | No | `YYYY-MM-DD` | Date of birth in Bikram Sambat. Derived by adding 57 years to AD year. For display in Nepali government documents. |
| `gender` | STRING | No | `male`, `female` | Applicant gender. Distribution: ~60% male, ~40% female (reflects rural Nepal credit-seeking patterns). |
| `marital_status` | STRING | No | `Married`, `Single`, `Widowed`, `Divorced` | Marital status. Distribution: ~68% Married, ~24% Single, ~6% Widowed, ~2% Divorced. |
| `province_en` | STRING | No | One of 7 provinces | Province of permanent address. Valid values: `Koshi`, `Madhesh`, `Bagmati`, `Gandaki`, `Lumbini`, `Karnali`, `Sudurpashchim`. **Noise:** ~2.5% have wrong province-district-municipality combinations. |
| `district_en` | STRING | No | One of 35 districts in dataset | District of permanent address (e.g. `Kavrepalanchok`, `Jhapa`, `Kaski`). Must be within the stated province (except geo-noise records). |
| `municipality_en` | STRING | No | Municipality/Sub-metro/Metro name | Smallest administrative unit (e.g. `Banepa Municipality`, `Pokhara Metropolitan City`, `Butwal Sub-Metropolitan City`). Must be within the stated district (except geo-noise records). |
| `ward_no` | INTEGER | Yes | 1 to max ward of municipality | Ward number within municipality. Nepal's municipalities are divided into wards (e.g. Kathmandu Metropolitan has 32 wards). **Noise:** ~2.5% are NULL (missing); ~1.2% are `0` (invalid/impossible). |
| `phone_primary` | STRING | Yes | 10-digit integer string | Synthetic phone number. **Deliberately does NOT start with 97, 98, or 99** (unlike real Nepal numbers) to ensure no real phone numbers are generated. First digit: 1–6. **Noise:** ~3% are NULL; ~4% are wrong length (7 or 12 digits). |
| `occupation_en` | STRING | No | See enum below | Primary occupation of the applicant. Key for income signal interpretation. |
| `occupation_np` | STRING | No | Nepali text | Occupation in Nepali (Devanagari script). Same data as `occupation_en`. |
| `education_level` | STRING | Yes | See enum below | Highest education level completed. **Noise:** ~12% are NULL (field not disclosed or not filled). |
| `household_size` | INTEGER | No | 1–8 | Number of people in the applicant's household. Relevant for per-capita income and loan serviceability calculations. |
| `land_area_ropani` | INTEGER | No | 0–21 | Area of agricultural/residential land owned, measured in Ropani (hill unit). `0` = no land. **Note:** Terai (plains) districts use Bigha, but this dataset standardizes to Ropani for simplicity. |
| `has_esewa_account` | BOOLEAN | No | `True`, `False` | Whether the applicant has a registered eSewa mobile wallet. ~58% true. |
| `has_khalti_account` | BOOLEAN | No | `True`, `False` | Whether the applicant has a registered Khalti mobile wallet. ~32% true. |
| `esewa_account_id` | STRING | Yes | `ESW-AP-NNNNNN` | eSewa wallet identifier. NULL if `has_esewa_account = False`. Used to link to `mobile_money_transactions`. |
| `khalti_account_id` | STRING | Yes | `KHL-AP-NNNNNN` | Khalti wallet identifier. NULL if `has_khalti_account = False`. |
| `primary_bank` | STRING | No | Bank name | Name of the applicant's primary bank account holder. One of 12 Nepali commercial banks. |
| `remittance_receiving` | BOOLEAN | No | `True`, `False` | Whether applicant receives international remittances from a family member working abroad. ~36% true. Rows with `True` will have records in `remittance_records`. |
| `cooperative_member` | BOOLEAN | No | `True`, `False` | Whether applicant is a registered member of an agricultural or savings cooperative. ~42% true. **Note:** ~3% have `True` here but no matching row in `cooperative_members` (cross-table noise — ghost membership). |
| `cooperative_id` | STRING | Yes | `COOP-XXX-NNNN` | Cooperative registration identifier. NULL if `cooperative_member = False`. |
| `kyc_tier` | STRING | No | `basic`, `mid`, `full` | KYC (Know Your Customer) tier completed at the bank. Determines maximum loan amount per NRB directives. See §5 for definition. |
| `rural_urban` | STRING | No | `rural`, `semi_urban`, `urban` | Settlement classification of applicant's address. Influences risk scoring and loan product eligibility. |
| `profile_created_date` | STRING | No | `YYYY-MM-DD` | Date the applicant profile was created in the system (AD format). |

**occupation_en valid values:**
`Farmer`, `Daily Wage Worker`, `Small Trader`, `Service Worker`, `Remittance Dependent`,
`Artisan`, `Government Employee`, `Business Owner`, `Teacher`, `Driver`, `Nurse/Health Worker`

**education_level valid values:**
`Illiterate`, `Primary`, `SLC`, `Intermediate`, `Bachelors`, `Masters`, `PhD`

---

### 3.2 `loan_applications.csv`

**Description:** One loan application per customer. Contains the complete record of a
loan request — from what was applied for, through the agent pipeline outputs (income
estimate, credit score, compliance check), to the final lending decision.

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `application_id` | STRING | No | `LA-2081-NNNNNN` | Unique application identifier. Format: `LA-` + BS year + `-` + 6-digit serial. |
| `applicant_id` | STRING | No | FK to applicant_profiles | Foreign key linking to the applicant's profile. |
| `application_date_ad` | STRING | No | `YYYY-MM-DD` | Date the application was submitted (Gregorian). All records: `2024-04-15`. |
| `application_date_bs` | STRING | No | `YYYY-MM-DD` in BS | Same date in Bikram Sambat (`2081-01-02`). |
| `loan_purpose` | STRING | No | See enum below | Category of loan purpose. Determines applicable NRB sector limits and interest rate corridors. |
| `requested_amount_nrs` | MIXED | No | Integer or string | Loan amount requested in Nepali Rupees (NRs). **Noise:** ~3% are stored as strings `"Rs. 200,000"` instead of numeric; ~0.5% are negative (data entry error). Participants must handle type coercion. |
| `requested_tenure_months` | INTEGER | Yes | 6, 12, 18, 24, 36 | Requested repayment period in months. **Noise:** ~4% are NULL. |
| `province_en` | STRING | No | Province name | Province of applicant (replicated from profile for query convenience). |
| `district_en` | STRING | No | District name | District of applicant. |
| `municipality_en` | STRING | No | Municipality name | Municipality of applicant. |
| `ward_no` | INTEGER | Yes | 1–max ward | Ward number (may be NULL/0 if profile has ward noise). |
| `rural_urban` | STRING | No | `rural`, `semi_urban`, `urban` | Settlement type of applicant's address. |
| `collateral_type` | STRING | No | `land`, `none` | Type of collateral offered. `land` = Lalpurja (land deed) submitted as security. NRB requires collateral for loans above NRs 5,00,000. |
| `collateral_value_nrs` | INTEGER | No | 0 or positive integer | Estimated market value of collateral in NRs. `0` if no collateral. |
| `has_esewa_account` | BOOLEAN | No | `True`, `False` | Replicated from applicant_profiles for join-free access. |
| `has_khalti_account` | BOOLEAN | No | `True`, `False` | Replicated from applicant_profiles. |
| `remittance_receiving` | BOOLEAN | No | `True`, `False` | Replicated from applicant_profiles. |
| `cooperative_member` | BOOLEAN | No | `True`, `False` | Replicated from applicant_profiles. |
| `cooperative_id` | STRING | Yes | `COOP-XXX-NNNN` | Cooperative ID if member, else NULL. |
| `existing_loan_count` | INTEGER | No | 0–3 | Number of active loans the applicant already has at other institutions. Relevant for NRB single-borrower exposure limits. |
| `credit_bureau_score` | FLOAT | Yes | 300–850 or NULL | Credit bureau score from CIB (Credit Information Bureau of Nepal). **NULL for ~72% of applicants** — the majority of target population has no formal credit history. This is the core problem the dataset addresses. |
| `nrb_blacklist_flag` | BOOLEAN | No | `True`, `False` | Whether the applicant appears on NRB's blacklist of defaulters. ~0.5% true. Applications with `True` must be rejected per NRB Unified Directives 2080. |
| `aml_flag` | BOOLEAN | No | `True`, `False` | Anti-Money Laundering screening flag. `True` if applicant matches patterns associated with money laundering. ~0.8% true. Triggers mandatory compliance review. |
| `doc_completeness_score` | FLOAT | Yes | 0.0–1.0 or NULL | Proportion of required documents successfully submitted and verified (0 = none, 1 = all). **Noise:** ~5% are NULL. |
| `income_agent_monthly_est` | INTEGER | No | 3,000–200,000 | Monthly income estimate (NRs) generated by the Income Agent from alternative data signals (eSewa transactions, remittances, cooperative sales). This is a **model output**, not self-reported income. |
| `income_confidence` | FLOAT | No | 0.05–0.97 | Confidence score (0–1) of the Income Agent's estimate. Low confidence (<0.40) triggers conservative lending decisions. Affected by data sparsity — applicants with only one income signal get lower confidence. |
| `credit_score` | INTEGER | No | 300–850 | Alternative credit score generated by the Score Agent using XGBoost on non-traditional data signals. Scale mirrors standard credit bureau scoring for interpretability. |
| `score_band` | STRING | No | See enum below | Categorical band derived from `credit_score`. |
| `compliance_status` | STRING | No | `pass`, `flag` | Result of NRB Compliance Agent check. `flag` = one or more NRB rules raised a soft warning (amount adjusted, referral required). No records have `veto` in this sample (hard vetoes lead to `final_decision = reject`). |
| `compliance_flags` | STRING | No | JSON array or `[]` | Array of compliance rule codes violated (e.g. `["LTI_EXCEEDED"]`). Empty array `[]` for clean applications. |
| `final_decision` | STRING | Yes | See enum below | Final lending decision from the Decision Agent. **Noise:** ~1.5% are NULL (system error / decision not recorded). |
| `approved_amount_nrs` | FLOAT | Yes | Positive integer or NULL | Amount approved for disbursement (NRs). NULL if decision is `refer` or `reject`. May differ from `requested_amount_nrs` when income confidence is low (haircut applied). |
| `interest_rate_pct` | MIXED | No | 10.0–15.0 or 0.0 | Annual interest rate (%) for approved loans. `0.0` for rejected/referred applications. NRB corridor for agricultural loans: 10%–15%. **Noise:** ~2% are outside corridor (16–22%) — NRB rule violation to detect. |
| `interest_tier` | STRING | No | `base`, `premium`, `subprime` | Pricing tier based on credit score and risk profile. Determines interest rate range. |
| `processing_time_seconds` | INTEGER | No | 12–200 | Time taken by the multi-agent orchestrator to process the application end-to-end. |
| `data_split` | STRING | No | `train`, `val`, `public_test` | Dataset partition. Distribution: 70% train, 25.6% val, 4.5% public_test. A hidden `hidden_test` partition exists with labels removed. |

**loan_purpose valid values:**
`agricultural_input` (28%), `small_trade` (14%), `livestock_purchase` (12%), `home_repair` (10%),
`microenterprise_startup` (9%), `education` (8%), `agri_land_development` (7%),
`irrigation_equipment` (5%), `vehicle_purchase` (4%), `emergency_medical` (3%)

**score_band valid values:**
`poor` (300–449), `fair` (450–579), `good` (580–669), `very_good` (670–739), `excellent` (740–850)

**final_decision valid values:**
`approve` (29%), `refer` (33%), `conditional` (19%), `reject` (18%), NULL (1.5% — noise)

**interest_tier valid values:**
`base` (< 11.5% — best credit score), `premium` (11.5%–13%), `subprime` (> 13%)

---

### 3.3 `mobile_money_transactions.parquet`

**Description:** Individual eSewa and Khalti mobile wallet transactions for applicants
who have a mobile wallet (~69% of customers). Covers 6 months of transaction history
(January–June 2024). The primary source of alternative income signals for the Income Agent.
Also contains injected fraud patterns for anomaly detection tasks.

> **Load with:** `pd.read_parquet("structured/mobile_money_transactions.parquet")`

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `transaction_id` | STRING | No | `TX-AP-NNNNNN-MMKK` | Unique transaction identifier. Format: `TX-` + applicant_id + `-` + month (2 digits) + sequence (2 digits). |
| `applicant_id` | STRING | No | FK to applicant_profiles | Links transaction to the wallet holder. |
| `platform` | STRING | No | `esewa`, `khalti` | Mobile wallet platform used. If applicant has both wallets, `esewa` is preferred. |
| `transaction_date` | STRING | No | `YYYY-MM-DD HH:MM:SS` | Timestamp of transaction (Nepal Standard Time, UTC+5:45). **Noise:** ~0.8% have future dates (2026) — impossible timestamps. |
| `transaction_type` | STRING | No | See enum below | Category of transaction. Critical for income signal extraction — `remittance_receipt` and `wallet_topup` are income proxies; `utility_payment` is a creditworthiness proxy. |
| `amount_nrs` | STRING | No | Numeric string or comma-formatted | Transaction amount in NRs. Stored as STRING because noise injects comma-formatted values (`"1,500.00"`). Parse with `pd.to_numeric(df.amount_nrs.str.replace(",",""), errors="coerce")`. |
| `direction` | STRING | No | `credit`, `debit` | Flow of funds: `credit` = money received into wallet; `debit` = money spent or withdrawn. |
| `counterparty_category` | STRING | Yes | See enum below | Category of the merchant or counterparty. **Noise:** ~4% are NULL (counterparty not recorded). |
| `geolocation_district` | STRING | No | District name | District where transaction was initiated. Useful for geographic spending pattern analysis. |
| `is_festival_period` | BOOLEAN | No | `True`, `False` | Whether the transaction occurred during a major Nepali festival period (Dashain/Tihar in months 3–4). Festival periods show elevated spending. |
| `_noise_anomaly_flag` | BOOLEAN | No | `True`, `False` | `True` if this transaction has an injected anomaly (fraud pattern). ~1.7% of transactions. See §4. |
| `_noise_anomaly_type` | STRING | Yes | `midnight_round`, `future_timestamp`, `velocity_burst`, NULL | Type of anomaly injected. NULL for clean transactions. Hidden in test split. |
| `_noise_null_party` | BOOLEAN | No | `True`, `False` | `True` if `counterparty_category` was deliberately set to NULL. |
| `_noise_amount_string` | BOOLEAN | No | `True`, `False` | `True` if `amount_nrs` was stored as comma-formatted string (`"1,500.00"`) rather than numeric. |

**transaction_type valid values:**
`merchant_payment` (28%), `p2p_transfer` (22%), `utility_payment` (18%), `qr_payment` (10%),
`remittance_receipt` (10%), `wallet_topup` (7%), `wallet_withdrawal` (3%), `loan_repayment` (2%)

**counterparty_category valid values:**
`grocery`, `utility`, `telecom`, `agriculture_input`, `medical`, `transport`,
`financial_services`, `remittance_agent`, `education`, `restaurant`

---

### 3.4 `remittance_records.parquet`

**Description:** International money transfer records for applicants who receive
remittances from family members working abroad (~36% of customers). Nepal is one of the
world's top remittance-receiving countries (remittances = ~25% of GDP). Each row is one
transfer event. Multiple transfers per applicant over the 6-month period.

> **Load with:** `pd.read_parquet("structured/remittance_records.parquet")`

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `remittance_id` | STRING | No | `REM-AP-NNNNNN-MMDD-TT` | Unique transfer identifier. |
| `applicant_id` | STRING | No | FK to applicant_profiles | Receiver's applicant ID. |
| `receiver_name_en` | STRING | No | Full name | Receiver's name as registered at the remittance agent (may differ from applicant_profiles name due to noise). |
| `receiver_district_en` | STRING | No | District name | Receiver's district (matches profile for clean records). |
| `receiver_municipality_en` | STRING | No | Municipality name | Receiver's municipality. |
| `sender_name` | STRING | No | Full name | Sender's name as declared at originating country branch. **Noise:** ~12% have last 2 characters dropped (OCR/transcription error simulation) — `name_match_score` < 0.80 for these. |
| `sender_country_code` | STRING | No | ISO 3166-1 alpha-2 | 2-letter country code of origin (e.g. `QA`=Qatar, `SA`=Saudi Arabia, `AE`=UAE). |
| `sender_country_name` | STRING | No | Country name | Full country name. |
| `sender_city` | STRING | No | City name | City where sender initiated transfer. |
| `transfer_service` | STRING | No | See enum below | Remittance service provider used. |
| `transfer_date_ad` | STRING | No | `YYYY-MM-DD` | Date sender initiated transfer at origin. |
| `received_date_ad` | STRING | No | `YYYY-MM-DD` | Date funds were made available to receiver in Nepal. Usually transfer_date + 1 business day. |
| `amount_foreign_currency` | FLOAT | No | 100.00–900.00 | Amount transferred in sender's currency. |
| `foreign_currency_code` | STRING | No | ISO 4217 currency code | Currency of the sending country. **Noise:** ~2.5% are `USD` regardless of actual country (wrong currency — data quality error). |
| `amount_nrs` | FLOAT | No | Positive float | Amount received in Nepali Rupees (after currency conversion). Calculated as `amount_foreign_currency × exchange_rate`. |
| `exchange_rate` | FLOAT | No | Positive float | Exchange rate applied at time of transfer (foreign currency per 1 NRs). **Noise:** ~2% have rate 10× the actual rate (impossible exchange rate — outlier to detect). |
| `transaction_fee_nrs` | FLOAT | No | 50.0 | Fee charged by the remittance service in NRs. Fixed at 50 in this dataset. |
| `disbursement_mode` | STRING | No | See enum below | How the receiver collected funds. |
| `relationship_to_receiver` | STRING | No | Family relationship | Sender's relationship to receiver. |
| `purpose_declared` | STRING | No | `family_support` | Stated purpose of transfer (all records: `family_support` in this dataset). |
| `name_match_score` | FLOAT | No | 0.0–1.0 | Fuzzy string match score between `sender_name` and the applicant's known family member name. Score < 0.80 should trigger a soft flag for manual review. Corrupted names score ~0.74. |
| `_noise_name_corrupt` | BOOLEAN | No | `True`, `False` | `True` if sender_name has last 2 characters dropped. |
| `_noise_wrong_currency` | BOOLEAN | No | `True`, `False` | `True` if foreign_currency_code is wrong (`USD` instead of correct currency). |
| `_noise_impossible_rate` | BOOLEAN | No | `True`, `False` | `True` if exchange_rate is 10× actual rate. |
| `_noise_duplicate` | BOOLEAN | No | `True`, `False` | `True` if this row is an exact duplicate of the preceding row. |

**transfer_service valid values:** `IME Money`, `Prabhu Money`, `Himal Remit`, `Western Union NP`

**disbursement_mode valid values:** `bank_deposit` (65%), `mobile_wallet` (25%), `cash_pickup` (10%)

**Sender country distribution:** Qatar (38%), Saudi Arabia (24%), UAE (14%), Malaysia (11%), India (7%), South Korea (3%), USA (2%), Japan (1%)

---

### 3.5 `utility_payments.parquet`

**Description:** Monthly electricity and mobile phone bill payment records for applicants.
Covers 6 billing months. Nepal Electricity Authority (NEA) electricity bills and Ncell
mobile bills are tracked. Payment regularity (`cumulative_on_time_rate`) is a proven proxy
for creditworthiness in alternative lending models.

> **Load with:** `pd.read_parquet("structured/utility_payments.parquet")`

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `payment_id` | STRING | No | `UTIL-AP-NNNNNN-ELE-MM` | Unique payment record ID. Suffix: `ELE` for electricity, `mob` for mobile. |
| `applicant_id` | STRING | No | FK to applicant_profiles | Links to the bill payer. |
| `province_en` | STRING | No | Province name | Province of service address. |
| `district_en` | STRING | No | District name | District of service address. |
| `municipality_en` | STRING | No | Municipality name | Municipality of service address. |
| `ward_no` | FLOAT | Yes | 1–max ward or NULL | Ward of service address (may be NULL from profile noise). |
| `utility_type` | STRING | No | `electricity`, `mobile` | Type of utility service. |
| `provider` | STRING | No | `NEA`, `Ncell` | Service provider. NEA = Nepal Electricity Authority (government monopoly electricity provider). Ncell = leading private mobile operator. |
| `service_number` | STRING | No | `XXX-NNNNN` | Consumer/account number with the utility provider. Format: district abbreviation + 5-digit account number. |
| `billing_period_bs` | STRING | No | `YYYY-MM` in BS | Billing period in Bikram Sambat (e.g. `2081-01` = first month of BS year 2081). |
| `billing_period_ad` | STRING | No | `YYYY-MM` | Same billing period in AD (e.g. `2024-01`). |
| `bill_amount_nrs` | FLOAT | No | Usually 150–5000 | Bill amount due in NRs. **Noise:** ~0.8% are negative (data entry error). For electricity, correlates with `units_consumed × 7.30`. |
| `units_consumed` | FLOAT | Yes | 0.0 or positive, or NULL | Electricity units consumed in kWh. NULL for mobile bills. **Noise:** ~1.2% of electricity records have `units_consumed = 0.0` despite a non-zero bill (meters not read / data error). |
| `due_date_ad` | STRING | No | `YYYY-MM-28` | Bill payment due date. NEA/Ncell bills are due on the 28th of each month. |
| `payment_date_ad` | STRING | Yes | `YYYY-MM-DD` or NULL | Actual payment date. NULL if unpaid. |
| `payment_method` | STRING | Yes | `esewa`, `khalti`, `bank`, or NULL | Payment channel used. NULL if unpaid. |
| `days_late` | FLOAT | Yes | Negative to positive integer, or NULL | Days relative to due date: negative = paid early, 0 = on time, positive = paid late, NULL = unpaid. |
| `cumulative_on_time_rate` | FLOAT | No | 0.30–1.0 | Running proportion of bills paid on or before due date for this customer-utility combination. Key creditworthiness signal. Higher = more reliable payer. |
| `outstanding_arrears_nrs` | FLOAT | No | 0.0 or positive | Total unpaid amount in NRs. Non-zero if `payment_date_ad` is NULL. |
| `_noise_forced_unpaid` | BOOLEAN | No | `True`, `False` | `True` if this bill was artificially set to unpaid (noise injection). |
| `_noise_negative_bill` | BOOLEAN | No | `True`, `False` | `True` if `bill_amount_nrs` is negative. |
| `_noise_zero_units` | BOOLEAN | No | `True`, `False` | `True` if `units_consumed = 0` despite non-zero bill. |
| `_noise_duplicate` | BOOLEAN | No | `True`, `False` | `True` if this row is an exact duplicate of the preceding row. |

---

### 3.6 `cooperative_members.csv`

**Description:** Membership records for applicants who belong to a registered agricultural
or savings cooperative (~41% of customers). Cooperatives are a critical alternative income
channel in rural Nepal — members sell produce through cooperatives, receive annual dividends,
and can access cooperative loans. **~3% of applicants have `cooperative_member = True` in
their profile but NO row here (ghost membership noise — a cross-table integrity error).**

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `member_id` | STRING | No | `CMEM-AP-NNNNNN` | Unique membership record ID. |
| `applicant_id` | STRING | No | FK to applicant_profiles | Links to the member's profile. |
| `cooperative_id` | STRING | No | `COOP-XXX-NNNN` | Cooperative registration ID. Format: `COOP-` + district abbreviation + `-` + 4-digit serial. |
| `cooperative_name_en` | STRING | No | Free text | Full name of cooperative in English (e.g. `Kavrepalanchok Krishak Sahakari Sanstha Ltd.`). |
| `cooperative_name_np` | STRING | No | Nepali text | Full name in Devanagari. |
| `cooperative_type` | STRING | No | See enum below | Type of cooperative by primary activity. Determines commodity types in `cooperative_sales`. |
| `province_en` | STRING | No | Province name | Province where cooperative is registered (same as member's province). |
| `district_en` | STRING | No | District name | District of cooperative registration. |
| `municipality_en` | STRING | No | Municipality name | Municipality of cooperative office. |
| `membership_year_bs` | INTEGER | No | 2072–2080 | Year of joining (Bikram Sambat). Tenure = current year minus this value. Longer tenure = stronger creditworthiness signal. |
| `share_count` | INTEGER | No | 5–50 | Number of cooperative shares purchased by member. Each share = NRs 1,000 par value. |
| `share_value_each_nrs` | INTEGER | No | 1000 | Par value per share (NRs). Fixed at 1,000 for all cooperatives in this dataset. |
| `total_share_value_nrs` | INTEGER | No | 5,000–50,000 | Total investment in cooperative shares (`share_count × share_value_each_nrs`). |
| `last_annual_dividend_nrs` | INTEGER | No | Positive integer | Dividend paid to member in the most recent fiscal year. Calculated as `share_count × share_value × dividend_rate`. A positive dividend indicates a financially healthy cooperative. |
| `outstanding_loan_nrs` | FLOAT | No | 0 or positive | Amount the member currently owes the cooperative as a borrowing member. Non-zero triggers compliance review for additional bank loans (double-leveraging check). |
| `coop_loan_repayment_status` | STRING | No | `none`, `current`, `overdue` | Status of any loan from the cooperative. `none` = no coop loan. `overdue` = negative creditworthiness signal. |
| `membership_status` | STRING | No | `active`, `inactive`, `suspended` | Current standing. `active` members can sell through cooperative and access dividends. `suspended` = disciplinary action (negative signal). |

**cooperative_type valid values:**
`agricultural` (42%), `dairy` (22%), `savings_credit` (20%), `vegetable` (10%), `coffee_tea` (6%)

---

### 3.7 `cooperative_sales.csv`

**Description:** Individual commodity sale records made by cooperative members through
their cooperative. Each row is one sale transaction (one commodity type, one season).
A member may have 0–3 sale records. Used to estimate agricultural income.
**Members of `savings_credit` cooperatives have no sale records** (savings cooperatives
handle deposits/loans, not commodity trading).

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `sale_id` | STRING | No | `CSALE-AP-NNNNNN-SS` | Unique sale record ID. |
| `applicant_id` | STRING | No | FK to applicant_profiles | Links to the selling member. |
| `cooperative_id` | STRING | No | FK to cooperative_members | Cooperative through which the sale was made. |
| `cooperative_type` | STRING | No | See 3.6 enum | Replicated from cooperative_members for convenience. |
| `province_en` | STRING | No | Province name | Province of sale (same as cooperative's province). |
| `district_en` | STRING | No | District name | District of cooperative. |
| `municipality_en` | STRING | No | Municipality name | Municipality of cooperative. |
| `sale_year_bs` | INTEGER | No | 2078–2081 | Bikram Sambat year of sale. |
| `season` | STRING | No | `Kharif`, `Rabi`, `Annual` | Agricultural season of sale. Kharif = monsoon crop (paddy, June–Nov). Rabi = winter crop (wheat, vegetables, Oct–Mar). Annual = year-round (dairy). |
| `commodity_en` | STRING | No | Commodity name | Name of commodity sold. Varies by cooperative type. |
| `unit` | STRING | No | `kg`, `L` | Unit of measurement. `kg` for solid commodities, `L` for milk. |
| `quantity` | INTEGER | No | 50–1500 | Quantity sold in stated unit. |
| `rate_nrs_per_unit` | INTEGER | No | Positive integer | Price per unit received (NRs). Set by cooperative based on market rates. |
| `total_amount_nrs` | INTEGER | No | Positive integer | Total sale amount (`quantity × rate_nrs_per_unit`). Primary income signal from cooperative channel. |

**Commodity by cooperative type:**
- `agricultural`: Paddy (NRs 35/kg), Vegetables (NRs 42/kg), Maize (NRs 28/kg), Wheat (NRs 32/kg)
- `dairy`: Milk (NRs 90/L), Ghee (NRs 1200/kg), Curd (NRs 180/kg)
- `vegetable`: Tomato (NRs 55/kg), Potato (NRs 28/kg), Cabbage (NRs 35/kg), Onion (NRs 60/kg)
- `coffee_tea`: Coffee Beans (NRs 400/kg), Tea Leaves (NRs 120/kg)
- `savings_credit`: No commodity sales

---

### 3.8 `document_registry.csv`

**Description:** Registry of all physical and scanned documents submitted by applicants
as part of the loan application. Each row is one document. A customer may have up to 6
different document types. Document images are referenced by `file_path` and are available
separately. Ground truth OCR annotations are in `ground_truth/{document_type}/{doc_id}_gt.json`.

| Column | Type | Nullable | Valid Values / Format | Description |
|---|---|---|---|---|
| `document_id` | STRING | No | `DOC-XXX-NNNNNNN` | Unique document identifier. Format: `DOC-` + type abbreviation + `-` + 7-digit serial. |
| `applicant_id` | STRING | No | FK to applicant_profiles | Owner of the document. |
| `document_type` | STRING | No | See enum below | Category of document. |
| `document_subtype` | STRING | No | Sub-classification | Further classification within type. |
| `file_path` | STRING | No | `docs/{type}/{doc_id}.jpg` | Relative path to document image within the dataset package. |
| `file_format` | STRING | No | `jpg` | Image file format. All documents are JPEG. |
| `page_count` | INTEGER | No | 1–2 | Number of pages. Cooperative passbooks are 2-page. All other documents are 1-page. |
| `scan_dpi` | INTEGER | No | 72, 96, 150, 200, 300 | Resolution of scan/photo. Low DPI (72–96) indicates mobile camera photos. High DPI (200–300) indicates flatbed scanner. |
| `ocr_complexity_tag` | STRING | No | See enum below | OCR difficulty classification. Determines expected OCR error rate. |
| `language_primary` | STRING | No | `nepali`, `english` | Primary language of document content. |
| `language_secondary` | STRING | Yes | `nepali`, `english`, or NULL | Secondary language (for bilingual documents). NULL for Lalpurja (Nepali only). |
| `has_stamp` | BOOLEAN | No | `True`, `False` | Whether document has an official government or bank rubber stamp. |
| `has_signature` | BOOLEAN | No | `True`, `False` | Whether document has handwritten signatures (officer and/or applicant). |
| `has_handwritten_fields` | BOOLEAN | No | `True`, `False` | Whether the document has fields filled in by hand (rather than printed). |
| `is_rotated` | BOOLEAN | No | `True`, `False` | Whether the document is rotated/skewed (common with mobile camera scans). ~12% of documents. |
| `rotation_angle_degrees` | FLOAT | No | -15.0 to +15.0 | Rotation angle in degrees. 0.0 if not rotated. Positive = clockwise. |
| `upload_date_bs` | STRING | No | `YYYY-MM-DD` in BS | Date document was uploaded to the system. |
| `ocr_model_baseline_cer` | FLOAT | No | 0.03–0.21 | Baseline Character Error Rate (CER) achieved by a standard OCR model (PaddleOCR v2.7) on this document. Lower = easier. Used to benchmark participant OCR solutions. |
| `verified_by_agent` | BOOLEAN | No | `True`, `False` | Whether the Document Agent successfully verified this document. ~5% are `False` (verification failed). |
| `verification_confidence` | FLOAT | No | 0.55–0.98 | Agent's confidence in document authenticity (0–1). |
| `anomaly_flag` | BOOLEAN | No | `True`, `False` | Whether the document contains an injected anomaly (e.g. altered field, mismatched name/photo). ~2.5% of documents. |
| `ground_truth_path` | STRING | No | `ground_truth/{type}/{doc_id}_gt.json` | Path to the JSON file containing OCR ground truth annotations for this document. |

**document_type valid values and prevalence:**
- `citizenship_certificate` (98% of applicants) — Nepali National Citizenship Certificate
- `utility_bill` (90%) — NEA electricity bill as proof of address
- `kyc_form` (85%) — Bank-issued Know Your Customer form
- `lalpurja` (65%) — Land ownership certificate (only land owners)
- `cooperative_passbook` (42%) — Cooperative membership passbook
- `remittance_receipt` (35%) — International money transfer receipt

**ocr_complexity_tag valid values:**
- `clean` — High-quality flatbed scan, good lighting, no obstructions. Baseline CER ~0.05
- `stamp_overlay` — Official stamp partially overlaps text fields. CER ~0.10
- `shadow` — Uneven lighting from mobile camera. CER ~0.12
- `blur` — Camera shake or out-of-focus. CER ~0.14
- `compression` — Heavy JPEG compression (WhatsApp-forwarded image). CER ~0.18

---

## 4. Noise Columns Reference

All noise indicator columns are prefixed `_noise_`. They are present in train/validation/public_test splits.
**In the hidden_test split, all `_noise_*` columns are removed** — participants must detect data quality
issues independently.

### 4.1 Purpose of Noise Columns

These columns serve two purposes:
1. **Validation:** During EDA, you can compare your cleaning pipeline output against these flags
2. **Training signal:** You may use these as additional features in models if desired

### 4.2 Noise Column Quick Reference

| Column | Table | Meaning | Rate |
|---|---|---|---|
| `_noise_geo_wrong` | profiles | Province/district/municipality is a wrong combination | 2.5% |
| `_noise_ward_issue` | profiles | Ward is NULL or 0 (impossible) | 3.6% |
| `_noise_dob_bad` | profiles | DOB is in the future or before 1924 | 1.3% |
| `_noise_name_format` | profiles | Name is ALL CAPS or all lower | 4.8% |
| `_noise_cit_bad` | profiles | Citizenship number has wrong format | 2.6% |
| `_noise_phone_bad` | profiles | Phone is NULL, 7 digits, or 12 digits | 6.7% |
| `_noise_edu_missing` | profiles | education_level is NULL | 11.9% |
| `_noise_duplicate` | profiles | This record is a near-duplicate | 1.5% |
| `_noise_amount_string` | loans | requested_amount_nrs is "Rs. 200,000" not numeric | 3% |
| `_noise_negative_amount` | loans | requested_amount_nrs is negative | 0.5% |
| `_noise_tenure_missing` | loans | requested_tenure_months is NULL | 4% |
| `_noise_decision_missing` | loans | final_decision is NULL | 1.5% |
| `_noise_interest_oor` | loans | interest_rate_pct > 15% (outside NRB corridor) | 2% |
| `_noise_duplicate_app` | loans | Same applicant has a duplicate application | 1.8% |
| `_noise_anomaly_flag` | transactions | Transaction is a fraud/anomaly pattern | 1.68% |
| `_noise_anomaly_type` | transactions | Type: midnight_round / future_timestamp / velocity_burst | — |
| `_noise_null_party` | transactions | counterparty_category is NULL | 4% |
| `_noise_amount_string` | transactions | amount_nrs is comma-formatted string | 3% |
| `_noise_name_corrupt` | remittance | sender_name has last 2 chars dropped | 12% |
| `_noise_wrong_currency` | remittance | foreign_currency_code is wrong | 2.5% |
| `_noise_impossible_rate` | remittance | exchange_rate is 10× actual | 2% |
| `_noise_duplicate` | remittance | Exact duplicate row | 2.5% |
| `_noise_forced_unpaid` | utility | Bill artificially set to unpaid | 15% |
| `_noise_negative_bill` | utility | bill_amount_nrs is negative | 0.8% |
| `_noise_zero_units` | utility | units_consumed=0 despite non-zero bill | 1.2% |
| `_noise_duplicate` | utility | Exact duplicate row | 1.5% |

---

## 5. Banking & Financial Terminology Glossary

This glossary defines all domain-specific banking, regulatory, and financial terms used
in the dataset and challenge.

---

**AML (Anti-Money Laundering)**
A set of laws, regulations, and procedures aimed at preventing criminals from disguising
illegally obtained funds as legitimate income. In Nepal, governed by the Asset (Money)
Laundering Prevention Act 2008 (amended). In this dataset, `aml_flag = True` indicates
the applicant triggered AML screening rules and requires manual review before any loan
approval.

---

**Alternative Credit Scoring**
Credit assessment methodology that uses non-traditional data signals — mobile money
transactions, remittance history, utility payment regularity, cooperative membership,
agricultural sales — instead of (or in addition to) formal credit bureau scores. Critical
for serving the ~55% of Nepalese adults who are "credit invisible" due to lack of formal
financial history.

---

**Arrears (बाँकी रकम)**
Overdue unpaid debt obligations. In `utility_payments`, `outstanding_arrears_nrs` captures
the total unpaid bill amount. Persistent arrears (missing payments across multiple months)
are a negative creditworthiness signal.

---

**Base Rate (आधार दर)**
The minimum benchmark interest rate set by NRB below which banks cannot offer loans.
Protects bank profitability and prevents predatory pricing.

---

**CER (Character Error Rate)**
OCR evaluation metric. Proportion of individual characters incorrectly recognized:
`CER = (Insertions + Deletions + Substitutions) / Total characters in ground truth`.
Lower is better. Industry benchmark for Devanagari OCR: < 5% for clean documents.

---

**CIB (Credit Information Bureau)**
Nepal's central credit registry, established under NRB mandate. Maintains records of
borrowers and their repayment history. `credit_bureau_score` in this dataset represents
a CIB score (range 300–850, higher = better). ~72% of dataset applicants have NULL
CIB scores — they have never taken a formal loan and therefore have no credit history.

---

**Collateral (धितो)**
Asset pledged by a borrower to secure a loan. In Nepal, agricultural land (evidenced by
Lalpurja) is the most common collateral for rural loans. In this dataset:
`collateral_type = "land"` means a Lalpurja was submitted. Per NRB rules, loans above
NRs 5,00,000 require registered collateral.

---

**Compliance Flag (अनुपालन चिह्न)**
In this dataset, a `compliance_flag` in `compliance_flags` column indicates that an NRB
rule was triggered. Format: JSON array of rule codes (e.g. `["LTI_EXCEEDED"]`).
`compliance_status = "flag"` means a soft warning — loan may proceed with adjustments.
`compliance_status = "pass"` means all NRB rules satisfied.

---

**Credit Score (साख अंक)**
A numerical representation of creditworthiness. Range 300–850 in this dataset (mirrors
FICO scale for interpretability). Generated by the Score Agent using XGBoost on alternative
data features. Bands: Poor (<450), Fair (450–579), Good (580–669), Very Good (670–739),
Excellent (740+).

---

**Data Split**
The partitioning of the dataset for machine learning:
- **Train (70%):** Labels visible. Use for model training.
- **Val (25.6%):** Labels visible. Use for hyperparameter tuning and validation.
- **Public Test (4.5%):** Labels visible at competition end (leaderboard scoring).
- **Hidden Test:** Labels never released. Final evaluation by organizers.

---

**Debit/Credit Direction**
In banking transaction records: `credit` means money flows **into** the account (income
signal); `debit` means money flows **out** (expense signal). Do not confuse with debit
cards — this is purely an accounting term.

---

**Disbursement Mode (भुक्तानी विधि)**
How loan proceeds or remittance funds are delivered to the recipient:
- `bank_deposit`: Transferred directly to a bank account
- `mobile_wallet`: Credited to eSewa/Khalti
- `cash_pickup`: Collected in cash at a remittance agent branch

---

**Dividend (लाभांश)**
Annual profit share distributed to cooperative members proportional to their shareholding.
A positive dividend (`last_annual_dividend_nrs > 0`) indicates the cooperative is
profitable — a positive signal for the member's financial ecosystem.

---

**Document Completeness Score**
`doc_completeness_score` in `loan_applications`. A value 0–1 representing the proportion
of required documents submitted and verified. Score < 0.70 typically triggers a `refer`
decision for human review. A score of NULL indicates the document verification pipeline
timed out or encountered an error.

---

**Exchange Rate (विनिमय दर)**
The rate at which one currency converts to another. In `remittance_records`:
`amount_nrs = amount_foreign_currency × exchange_rate`. Typical rates (2024 approximations):
QAR≈39 NRs, SAR≈35.5 NRs, AED≈36.8 NRs, MYR≈29.5 NRs, USD≈133.5 NRs, INR≈1.6 NRs.
**Noise:** ~2% of records have `exchange_rate` that is 10× actual — an outlier to detect.

---

**Financial Inclusion (वित्तीय समावेशीकरण)**
The goal of ensuring that individuals and businesses have access to useful and affordable
financial products and services. This dataset is specifically designed around the
financial inclusion challenge — extending credit to populations excluded from formal banking.

---

**Ghost Membership (भूत सदस्यता)**
A cross-table data integrity error in this dataset (and in real-world messy data):
`cooperative_member = True` in `applicant_profiles` but **no corresponding row** in
`cooperative_members`. Introduced in ~3% of cooperative-flagged applicants. Detecting
this is a data quality/EDA task.

---

**Haircut (कटौती)**
A reduction applied to a loan amount below the requested amount, due to risk factors.
In this dataset, when `income_confidence < 0.70`, the approved amount = requested × 0.85
(a 15% haircut). `approved_amount_nrs < requested_amount_nrs` signals a haircut was applied.

---

**Income Agent**
One of the five AI agents in the orchestration pipeline. Aggregates signals from eSewa/
Khalti transactions, remittance records, cooperative sales, and utility payment regularity
to estimate `income_agent_monthly_est` and `income_confidence`.

---

**Income Confidence**
`income_confidence` in `loan_applications`. A score 0–1 representing the Income Agent's
certainty in its monthly income estimate. Determined by:
- Number of independent income signals available (each signal adds confidence)
- Consistency of signals over time (low variance = higher confidence)
- Recency of data (older data = lower confidence)
Threshold: < 0.40 → reject; 0.40–0.70 → approve with haircut; > 0.70 → full approval.

---

**Interest Rate Corridor (ब्याजदर कोरिडोर)**
NRB sets minimum and maximum interest rates for each loan category. Agricultural loans:
10%–15% per annum. `interest_rate_pct` outside this range is an NRB rule violation.
In this dataset, ~2% of approved loans have rates outside the corridor (noise to detect).

---

**Interest Tier**
- `base`: Best rate (10.0%–11.5%). Credit score ≥ 670. Lowest risk.
- `premium`: Mid rate (11.5%–13.0%). Credit score 580–669. Moderate risk.
- `subprime`: Highest rate (13.0%–15.0%). Credit score < 580. High risk, requires justification.

---

**KYC (Know Your Customer) (ग्राहक पहिचान)**
Mandatory process for financial institutions to verify the identity of customers and
assess associated risks. Three tiers in Nepal:
- `basic`: Citizenship document only. Max loan: NRs 1,00,000.
- `mid`: Citizenship + utility bill (proof of address). Max loan: NRs 5,00,000.
- `full`: Full documentation (citizenship, address proof, income proof, photo). No loan ceiling.

---

**Lalpurja (लालपुर्जा)**
Official Nepali term for a Land Ownership Certificate issued by the District Land Revenue
Office (मालपोत कार्यालय). Full formal name: **जग्गाधनी दर्ता प्रमाण पुर्जा**
(Jagadhani Darta Pramaan Purja). The definitive proof of land ownership in Nepal, used as
loan collateral. Contains: owner name, address, plot number, land area (in Ropani/Aana/
Paisa/Daam for hills or Bigha/Kattha/Dhur for Terai), land type, estimated value, and
registration history. Referenced in this dataset by `DOC-LAL-XXXXXXX` in document_registry.

---

**LTI (Loan-to-Income Ratio)**
`requested_amount_nrs / income_agent_monthly_est`. NRB rule `NRB-LTI-002`: agricultural
loans must not exceed 36× monthly income. Triggering this rule shows as `LTI_EXCEEDED`
in `compliance_flags`.

---

**LTV (Loan-to-Value Ratio)**
`requested_amount_nrs / collateral_value_nrs`. Measures collateral adequacy. A ratio
> 0.80 is typically considered high risk (under-collateralized).

---

**Name Match Score**
`name_match_score` in `remittance_records`. A fuzzy string similarity score (0–1) between
the `sender_name` on the remittance receipt and the name of the known family member on
file. Score interpretation:
- ≥ 0.90: High confidence — same person
- 0.80–0.89: Soft match — possible OCR error or name variation
- < 0.80: Low match — flag for manual verification (corruption or different sender)

---

**Non-Performing Loan (NPL / खराब ऋण)**
A loan where the borrower has not made scheduled payments for 90+ days. NPL ratio is
a key banking health metric. Applicants with `coop_loan_repayment_status = "overdue"`
have existing non-performing cooperative loans — a strong negative credit signal.

---

**NRB (Nepal Rastra Bank)**
Nepal's central bank and primary banking regulator. Issues Unified Directives governing
all aspects of bank lending, KYC, AML, interest rates, collateral requirements, and
sector exposure limits. All decisions in this dataset must comply with NRB Unified
Directives 2080. Key rules modeled in the dataset:
- `NRB-KYC-001`: Minimum KYC tier required
- `NRB-LTI-002`: Loan-to-Income ratio cap
- `NRB-SEC-003`: Single-borrower sector exposure limit
- `NRB-AML-004`: AML blacklist check
- `NRB-COL-005`: Collateral requirement above NRs 5,00,000
- `NRB-DUP-006`: No duplicate applications within 90 days
- `NRB-INT-007`: Interest rate corridor compliance

---

**On-Time Payment Rate**
`cumulative_on_time_rate` in `utility_payments`. Running average of on-time payments
(days_late ≤ 0) for a customer-utility pair. Strong proxy for financial discipline:
- ≥ 0.90: Excellent payer — positive credit signal
- 0.70–0.89: Good payer — neutral
- 0.50–0.69: Moderate — mild negative signal
- < 0.50: Poor payer — significant negative signal

---

**Ropani (रोपनी)**
Unit of land measurement used in hilly regions of Nepal.
1 Ropani = 508.72 square meters ≈ 0.05 hectares.
1 Ropani = 16 Aana = 64 Paisa = 256 Daam.
Area format in Lalpurja: R-A-P-D (e.g. `0-2-3-0` = 0 Ropani 2 Aana 3 Paisa 0 Daam ≈ 57 sqm).
Note: Terai regions use Bigha (1 Bigha ≈ 0.68 hectares). This dataset uses Ropani for all regions.

---

**Sahakari (सहकारी) / Cooperative**
A member-owned business organization in Nepal. Agricultural cooperatives allow farmers to
collectively market produce, access inputs at bulk rates, and receive financing. Regulated
by the Department of Cooperatives under the Ministry of Land Management. In this dataset,
cooperatives are the third income channel (after mobile wallets and remittances) used by
the Income Agent.

---

**SHAP (SHapley Additive exPlanations)**
Machine learning interpretability technique. Assigns each feature a contribution value
(positive or negative) for a specific prediction. Required for all credit decisions in
this dataset — every approved/rejected application must have SHAP values attached in
the audit trail. Teams that provide SHAP explanations earn bonus marks.

---

**Score Agent**
AI agent in the pipeline that generates `credit_score` (300–850) using XGBoost trained
on normalized alternative data features. Outputs: credit_score, score_band, and SHAP
feature importance values.

---

**Velocity Burst (गति विस्फोट)**
A fraud pattern in mobile money transactions: multiple large transactions in a very
short time window (e.g. 10+ transactions within 5 minutes, or amount 8× user's
average). Flagged as `_noise_anomaly_type = "velocity_burst"` in transactions.

---

**WER (Word Error Rate)**
OCR evaluation metric at word level. `WER = (Word Substitutions + Deletions + Insertions) / Total words`.
Typically WER ≈ CER × 1.3 for Devanagari text. Benchmark target: WER < 10%.

---

## 6. Nepal-Specific Terms & Context

**Bikram Sambat (BS) Calendar**
The official calendar of Nepal. Approximately 56–57 years ahead of Gregorian (AD).
Example: AD 2024 ≈ BS 2081. Dates in this dataset appear in both formats.
Government documents (Lalpurja, citizenship) use BS exclusively.

**Baisakh (बैशाख)**
First month of the Bikram Sambat year (mid-April to mid-May). Nepal's New Year begins
in Baisakh. The Kharif planting season starts in Ashadh (3rd month), while the Rabi
harvest peaks in Chaitra/Baisakh (12th/1st months).

**Bikash Sewa Kendra / eSewa / Khalti**
Nepal's leading mobile payment platforms:
- **eSewa** (Fonepay): Largest mobile wallet (~58% of dataset users). Supports bill
  payments, merchant QR, P2P transfers, remittance receipt.
- **Khalti** (~32% users): Second-largest mobile wallet, popular with younger users.

**IME Money / Prabhu Money / Himal Remit**
Nepal's major domestic remittance aggregators and international money transfer companies.
IME Money: largest network (India corridor dominant). Prabhu Money: strong in Gulf corridor.
Himal Remit: popular for Korea/Japan corridors.

**जिल्ला प्रशासन कार्यालय (District Administration Office)**
The government office at the district level that issues citizenship certificates and
performs other administrative functions. The `citizenship_office` field contains these
office names.

**मालपोत कार्यालय (Land Revenue Office)**
The office that issues and maintains Lalpurja (land ownership certificates). Located
at district level. Also handles land transfer registration (राजिनामा).

**नेपाल विद्युत् प्राधिकरण — NEA (Nepal Electricity Authority)**
The government electricity utility. Near-monopoly provider. Monthly bills issued on the
Nepali billing cycle (BS months). Bill payments via eSewa/Khalti are a primary utility
payment signal. Consumer numbers follow district-based prefixes (e.g. KAV = Kavrepalanchok).

**राजिनामा (Rajinama)**
Land transfer deed — the legal document recording the sale/transfer of land ownership.
Referenced in Lalpurja transaction history columns (`transaction_type_np = "राजिनामा"`).

---

## 7. Data Formats & Conventions

| Aspect | Convention | Example |
|---|---|---|
| Primary Key | `TABLE_CODE-NNNNNN` | `AP-050234` |
| Application ID | `LA-YYYY-NNNNNN` | `LA-2081-050234` |
| Document ID | `DOC-XXX-NNNNNNN` | `DOC-LAL-0050234` |
| Transaction ID | `TX-AP-NNNNNN-MMKK` | `TX-AP-050234-0203` |
| Cooperative ID | `COOP-DDD-NNNN` | `COOP-KAV-0042` |
| Dates (AD) | ISO 8601 `YYYY-MM-DD` | `2024-07-30` |
| Dates (BS) | `YYYY/MM/DD` or `YYYY-MM-DD` | `2081/04/15` |
| Timestamps | `YYYY-MM-DD HH:MM:SS` | `2024-03-14 14:23:11` |
| Currency | NRs (Nepali Rupees) | `200000.00` |
| Phone | 10-digit string (first digit 1–6) | `3421987654` |
| Boolean | `True` / `False` | `True` |
| NULL handling | Empty cell in CSV, `NaN` in Parquet | — |
| Noise columns | Prefix `_noise_` | `_noise_geo_wrong` |
| Parquet reading | `pd.read_parquet(path)` | — |

### Loading Large Parquet Files

```python
import pandas as pd

# Full load (fits in RAM for 100k dataset)
tx   = pd.read_parquet("structured/mobile_money_transactions.parquet")
util = pd.read_parquet("structured/utility_payments.parquet")
rem  = pd.read_parquet("structured/remittance_records.parquet")

# Selective columns (faster for large files)
tx_income = pd.read_parquet(
    "structured/mobile_money_transactions.parquet",
    columns=["applicant_id","transaction_type","amount_nrs","direction","transaction_date"]
)

# Filter for one applicant
ap_tx = tx[tx["applicant_id"] == "AP-050234"]
```

### Handling Mixed-Type Amount Columns

```python
# Fix amount_nrs in transactions (may be "1,500.00" string)
tx["amount_nrs_clean"] = pd.to_numeric(
    tx["amount_nrs"].astype(str).str.replace(",", "").str.replace("Rs. ", ""),
    errors="coerce"
)

# Fix requested_amount_nrs in loans (may be "Rs. 200,000" string or negative)
loans["amount_clean"] = pd.to_numeric(
    loans["requested_amount_nrs"].astype(str).str.replace("Rs. ","").str.replace(",",""),
    errors="coerce"
).abs()   # take absolute value to fix negatives
```

---

## 8. Evaluation Metrics Reference

### 8.1 Credit Model Metrics

| Metric | Formula | Target | Notes |
|---|---|---|---|
| Decision Accuracy | Correct decisions / Total | > 0.78 | 4-class: approve/conditional/refer/reject |
| Macro F1 | Mean F1 across all 4 classes | > 0.72 | Handles class imbalance |
| Gini Coefficient | 2 × AUC − 1 | > 0.45 | Credit score discriminative power |
| NRB Compliance Rate | NRB-compliant decisions / Total | = 1.00 | **Non-negotiable — violations disqualify** |
| Income RMSE | √(mean squared error of income estimates) | < NRs 5,000 | vs. ground truth monthly income |

### 8.2 OCR / Document Metrics

| Metric | Formula | Target | Notes |
|---|---|---|---|
| CER | (S+D+I) / N_chars | < 5% | Character Error Rate |
| WER | (S+D+I) / N_words | < 10% | Word Error Rate |
| Field F1 | Precision × Recall / (P+R) per field | > 0.85 | Named entity extraction accuracy |
| Doc Classification Accuracy | Correct type / Total docs | > 0.95 | 6 document types |

### 8.3 Anomaly Detection Metrics

| Metric | Target | Notes |
|---|---|---|
| Fraud Detection AUC | > 0.80 | Hidden labels: midnight_round, velocity_burst |
| Precision at K=100 | > 0.60 | Top-100 flagged transactions |
| Cross-table Integrity F1 | > 0.75 | Detecting ghost memberships, duplicate records |

---

*This document is part of the GIBL AI/ML Hackathon 2026 — Track A dataset package.*

*Generated by the GIBL Hackathon Technical Committee, May 2026.*

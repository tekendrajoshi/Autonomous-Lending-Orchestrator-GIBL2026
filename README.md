# Global IME Bank Lending Orchestrator
Note: This is not the actual project repository and do not contain the dataset.

This repository is a curated, shareable project folder prepared for application and portfolio review. It contains the code, trained model artifacts, preprocessing notebooks/scripts, dataset files, frontend, backend, pipeline outputs, and documentation. The original private bank repository link is intentionally not included, in line with bank confidentiality requirements.

## Project Summary

The Lending Orchestrator is an end-to-end AI-assisted loan decisioning system for applicants with limited formal credit history. It combines alternative financial signals such as wallet transactions, remittance behavior, cooperative records, utility payment history, applicant profiles, and loan application data.

The system uses a multi-agent LangGraph pipeline:

1. Income Agent estimates monthly income and confidence from alternative data.
2. Score Agent creates a credit score and score band using trained ML artifacts, with fallback scoring logic.
3. Compliance Agent checks lending policy constraints such as affordability, collateral, AML, blacklist, document quality, and regulatory limits.
4. Decision Agent synthesizes the upstream results into `APPROVE`, `CONDITIONAL`, `REFER`, or `REJECT` decisions with rationale and audit trails.

The project also includes a FastAPI backend, a React/Vite dashboard, trained `.joblib` and `.pkl` model files, preprocessing scripts, notebooks, and batch evaluation CSVs.

## Repository Structure

```text
global-ime-bank-lending-orchestrator-clean/
├── README.md
├── .gitignore
├── backend/
│   ├── main.py
│   ├── run_pipeline.py
│   ├── concepts.py
│   ├── pre_compliance.py
│   ├── requirements.txt
│   ├── credit_score_model.joblib
│   ├── credit_score_model.pkl
│   ├── trained_income_model.pkl
│   └── orchestrator/
│       ├── pipeline.py
│       ├── state.py
│       ├── data_loader.py
│       ├── income_agent.py
│       ├── score_agent.py
│       ├── compliance_agent.py
│       └── decision_agent.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── src/
├── data/
│   ├── applicant_profiles.csv
│   ├── loan_applications.csv
│   ├── cooperative_members.csv
│   ├── cooperative_sales.csv
│   ├── mobile_money_transactions.parquet
│   ├── remittance_records.parquet
│   └── utility_payments.parquet
├── models/
│   ├── credit_score_model.joblib
│   ├── credit_score_model.pkl
│   └── trained_income_model.pkl
├── notebooks/
│   ├── orchestrator/
│   └── preprocessing/
├── preprocessing_scripts/
├── evaluation_results/
├── docs/
│   ├── DATASET_DESCRIPTION.md
│   └── PROJECT_DESCRIPTION_FOR_APPLICATION.md
└── original_notes/
    └── original_orchestrator_README.md
```

## Key Features

- Multi-agent loan decision pipeline built with LangGraph.
- FastAPI backend with health, upload, single-application, batch, result, and audit endpoints.
- React/Vite frontend dashboard for data upload, batch execution, decisions, and audit review.
- ML-based credit scoring model artifacts in `.joblib` and `.pkl` formats.
- Trained income estimation artifact in `.pkl` format.
- Alternative-data feature engineering for wallet activity, remittances, cooperatives, utilities, and profile records.
- Rule-based compliance and decision synthesis with audit trail generation.
- Batch result files for evaluation and demonstration.

## Backend Setup

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API while pointing it to the curated top-level data and result folders:

```bash
DATA_DIR=../data RESULTS_DIR=../evaluation_results python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend API:

- `GET /status`
- `POST /upload`
- `POST /process`
- `POST /batch`
- `GET /results`
- `GET /result/{application_id}`
- `GET /audit/{application_id}`
- Swagger UI: `http://localhost:8000/docs`

## Frontend Setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

The frontend expects the backend at:

```text
http://localhost:8000
```

## CLI Usage

The backend also includes a command-line runner:

```bash
cd backend
DATA_DIR=../data RESULTS_DIR=../evaluation_results python3 run_pipeline.py --mode batch --limit 100 --data-dir ../data --output ../evaluation_results/batch_results.csv
```

Single applicant example:

```bash
cd backend
python3 run_pipeline.py --mode single --applicant AP-050234 --data-dir ../data
```

## Dataset

See [docs/DATASET_DESCRIPTION.md](docs/DATASET_DESCRIPTION.md) for table-level details, row counts, column counts, and each dataset's role in the pipeline.

## Models

Model artifacts are stored in [models](models):

- `credit_score_model.joblib`: primary credit score model artifact.
- `credit_score_model.pkl`: fallback/alternate serialized credit score model artifact.
- `trained_income_model.pkl`: trained income estimation model artifact.

Copies of the same model files are also kept in [backend](backend) because the original backend code loads them by relative path from the backend package.

## Evaluation Outputs

Batch result CSVs are stored in [evaluation_results](evaluation_results). These files capture sample pipeline runs with fields such as final decision, approved amount, interest rate, credit score, score band, income estimate, compliance status, flags, decision rationale, processing time, and errors.

## Notes

- The copied source artifacts are preserved as-is from the hackathon folders.
- Generated dependency folders such as `node_modules` and Python `__pycache__` directories are not included.
- The backend compliance agent references an optional external `compliance_agent/` package. That package was not present in the provided source folder, so the included backend keeps its built-in fallback compliance path.
- The original README from the hackathon folder is preserved at [original_notes/original_orchestrator_README.md](original_notes/original_orchestrator_README.md).

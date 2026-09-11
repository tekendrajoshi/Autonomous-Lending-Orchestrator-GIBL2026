import pandas as pd
import xgboost as xgb
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# Define our clipboard structure
class AgentState(TypedDict):
    application_id: str
    applicant_id: str
    income_estimate_monthly: Optional[float]
    income_confidence: Optional[float]
    credit_score: Optional[int]
    score_band: Optional[str]
    compliance_status: Optional[str]
    compliance_flags: List[str]







# ================================================================================

## SCORE AGENT

# Load your pre-trained model globally once when the server boots up

model = xgb.Booster()
model.load_model("app/ml/weights/credit_model.json")
# (So you don't waste time reloading it on every API call)



# This is your unified feature engineering utility function
def engineer_online_features(applicant_id: str, income_metrics: dict) -> pd.DataFrame:
    """
    This function must replicate the exact same feature columns, 
    aggregations, and cleaning steps that your training dataset had!'''


    # 1. Read the clipboard to find out who we are scoring
    target_id = state["applicant_id"]
    
    # 2. SQL's Job: Fetch ONLY this applicant's mobile money records from Supabase
    # We pass the global 'engine' connection directly into pandas read_sql
    sql_query = f"SELECT * FROM mobile_money_transactions WHERE applicant_id = '{target_id}'"
    raw_tx_df = pd.read_sql(sql_query, con=engine)
    
    # 3. Pandas' Job: Replicate your ML feature engineering pipeline on this user's data

    """

    # 1. Fetch data from production tables for this specific profile
    # profile_df = pd.read_sql(f"SELECT * FROM production.applicant_profiles WHERE applicant_id='{applicant_id}'", db)
    # utility_df = pd.read_sql(f"SELECT * FROM production.utility_payments WHERE applicant_id='{applicant_id}'", db)
    
    # 2. Replicate transformations (e.g., calculate cumulative_on_time_rate)
    # on_time_rate = utility_df['days_late'].le(0).mean()
    
    # 3. Form a single-row DataFrame matching your model's expected training features
    feature_dict = {
        "income_estimate_monthly": income_metrics["income_estimate_monthly"],
        "income_confidence": income_metrics["income_confidence"],
        "cumulative_on_time_rate": 0.92,  # Example engineered value
        "land_area_ropani": 4,            # Extracted from profile
        "existing_loan_count": 1          # Extracted from CIB records
    }
    
    return pd.DataFrame([feature_dict])



def score_agent_node(state: AgentState) -> dict:
    # 1. Get IDs from clipboard
    target_applicant = state["applicant_id"]
    
    # 2. Package the context collected by the previous Income Agent
    income_metrics = {
        "income_estimate_monthly": state["income_estimate_monthly"],
        "income_confidence": state["income_confidence"]
    }
    
    # 3. Call the unified transformation layer to build a 1-row vector
    input_features_df = engineer_online_features(target_applicant, income_metrics)
    
    # 4. Wrap it into DMatrix (or whatever your model requires) and predict
    dmatrix = xgb.DMatrix(input_features_df)
    ml_prediction = credit_model.predict(dmatrix)[0]  # Outputs a continuous probability or score
    
    predicted_score = int(ml_prediction)  # Map it back to a standard integer score (e.g., 620)
    
    # 5. Map the score to a band
    if predicted_score >= 670: band = "very_good"
    elif predicted_score >= 580: band = "good"
    else: band = "poor"
    
    return {
        "credit_score": predicted_score,
        "score_band": band
    }

        

   


## INCOME AGENT

def income_agent_node(state: AgentState) -> dict:
    # 1. Look at the clipboard to find out who we are evaluating
    target_applicant = state["applicant_id"]
    
    # 2. Query Supabase for ONLY this applicant's alternative transaction data
    # (Simulated SQL/Pandas read for mobile_money_transactions, remittance_records, cooperative_sales)
    # df_wallet = pd.read_sql(f"SELECT * FROM staging.mobile_money_transactions WHERE applicant_id='{target_applicant}'", supabase_engine)
    # df_remit = pd.read_sql(f"SELECT * FROM staging.remittance_records WHERE applicant_id='{target_applicant}'", supabase_engine)
    
    # 3. Perform the aggregate engineering on this individual's data subset
    # e.g., sum up total credits over the last 6 months divided by 6
    calculated_income = 45000.0  # Result of your Pandas aggregations
    confidence = 0.85
    
    # 4. Return only what needs to be written to the clipboard
    return {
        "income_estimate_monthly": calculated_income,
        "income_confidence": confidence
    }




# COMPLIANCE AGENT
def compliance_agent_node(state: AgentState) -> dict:
    app_id = state["application_id"]
    
    # 1. Pull the user's specific loan application record from Supabase
    # app_row = supabase.table("loan_applications").select("*").eq("application_id", app_id).execute().data[0]
    requested_amount = 600000.0  # pulled from loan_applications.requested_amount_nrs
    interest_rate = 14.5        # pulled from loan_applications.interest_rate_pct
    
    # 2. Pull the metrics generated dynamically by the previous agents from the state
    estimated_monthly_income = state["income_estimate_monthly"]
    
    flags = []
    
    # Rule A: Debt-to-Income / Loan-to-Income Cap Check
    # (e.g., Monthly installment cannot exceed 50% of verified alternative income)
    estimated_monthly_installment = (requested_amount / 12)
    if estimated_monthly_installment > (0.50 * estimated_monthly_income):
        flags.append("LTI_EXCEEDED")
        
    # Rule B: Uncollateralized High-Value Check
    # (NRB requirements state micro-loans > NRs 500,000 require formal collateral land deeds / Lalpurja)
    # if requested_amount > 500000.0 and app_row['collateral_type'] == 'none':
    if requested_amount > 500000.0:
        flags.append("UNCOLLATERALIZED_HIGH_VALUE")
        
    status = "flag" if len(flags) > 0 else "pass"
    
    return {
        "compliance_status": status,
        "compliance_flags": flags
    }



# ===============================================================================
## CONNECTING EVERYTHING TOGETHER WITH LANGGRAPH

# 1. Initialize the State Graph
workflow = StateGraph(AgentState)

# 2. Add our desk workers (Nodes)
workflow.add_node("income_agent", income_agent_node)
workflow.add_node("score_agent", score_agent_node)
workflow.add_node("compliance_agent", compliance_agent_node)

# 3. Establish the strict routing pipeline sequence
workflow.set_entry_point("income_agent")         # Start at Income Desk
workflow.add_edge("income_agent", "score_agent") # Go to Scoring Desk next
workflow.add_edge("score_agent", "compliance_agent") # Go to Compliance Desk next
workflow.add_edge("compliance_agent", END)       # Finish the pipeline

# 4. Compile the application graph
loan_processing_pipeline = workflow.compile()


''' The Ultimate Database Commit

When the execution state hits END, you invoke a final script that reads the completed 
dictionary on your AgentState clipboard. 
It issues a single, unified UPDATE command back into the loan_applications table in Supabase
 for that application_id, writing back the income_agent_monthly_est, credit_score, 
 score_band, and compliance_flags. This completes the cycle and marks the record 
 processing as finished.'''
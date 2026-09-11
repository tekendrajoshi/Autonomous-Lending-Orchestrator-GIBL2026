"""
orchestrator/pipeline.py
=========================
Assembles the LangGraph StateGraph that connects all four agents in sequence:

    Income Agent
        ↓
    Score Agent
        ↓
    Compliance Agent
        ↓
    Decision Agent
        ↓
       END

The compiled graph is a callable that accepts an initial AgentState dict
and streams through each desk, writing outputs back to the shared clipboard.
"""

from __future__ import annotations

import time
from typing import Optional

from langgraph.graph import StateGraph, END

from orchestrator.state import AgentState
from orchestrator.income_agent import income_agent_node
from orchestrator.score_agent import score_agent_node
from orchestrator.compliance_agent import compliance_agent_node
from orchestrator.decision_agent import decision_agent_node


def _build_pipeline() -> object:
    """
    Constructs and compiles the LangGraph StateGraph.
    Called once at import time and cached as `loan_pipeline`.
    """
    workflow = StateGraph(AgentState)

    # ── Add desk workers (nodes) ──────────────────────────────────────────────
    workflow.add_node("income_agent",     income_agent_node)
    workflow.add_node("score_agent",      score_agent_node)
    workflow.add_node("compliance_agent", compliance_agent_node)
    workflow.add_node("decision_agent",   decision_agent_node)

    # ── Strict sequential routing ─────────────────────────────────────────────
    workflow.set_entry_point("income_agent")
    workflow.add_edge("income_agent",     "score_agent")
    workflow.add_edge("score_agent",      "compliance_agent")
    workflow.add_edge("compliance_agent", "decision_agent")
    workflow.add_edge("decision_agent",   END)

    return workflow.compile()


# ── Singleton compiled graph ───────────────────────────────────────────────────
loan_pipeline = _build_pipeline()


def run_single_application(initial_state: dict) -> dict:
    """
    Runs one loan application through the full pipeline.
    Times execution and returns the complete final state.
    """
    start = time.time()

    final_state = loan_pipeline.invoke(initial_state)

    elapsed = round(time.time() - start, 3)
    final_state["processing_time_seconds"] = elapsed

    return final_state


def run_batch(
    all_states: list[dict],
    max_workers: int = 1,
    progress_callback=None,
) -> list[dict]:
    """
    Processes a batch of applications sequentially (or with threading).
    
    Args:
        all_states:        List of initial AgentState dicts (one per application)
        max_workers:       Number of parallel workers (default 1 = sequential)
        progress_callback: Optional callable(current_idx, total, result_dict)
    
    Returns:
        List of final AgentState dicts in the same order as input.
    """
    results = []
    total = len(all_states)

    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, state in enumerate(all_states):
                future = executor.submit(run_single_application, state)
                futures_map[future] = i

        results_dict = {}
        for future in as_completed(futures_map):
            i = futures_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**all_states[i], "error_message": str(exc), "final_decision": "error"}
            results_dict[i] = result
            if progress_callback:
                progress_callback(len(results_dict), total, result)

        results = [results_dict[i] for i in range(total)]
    else:
        for i, state in enumerate(all_states):
            try:
                result = run_single_application(state)
            except Exception as exc:
                result = {**state, "error_message": str(exc), "final_decision": "error"}
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total, result)

    return results

# graph.py

from langgraph.graph import StateGraph
from typing import TypedDict
import pandas as pd

from agents.sample_gen_agent import generate_sample_data
from agents.gap_det_agent import clean_data
from agents.gan_agent import generate_tabular_data_ctgan
from agents.eda_agent import analyze_data
from agents.feed_agent import decide_regeneration

MAX_RETRIES = 2

class AppState(TypedDict):
    df: pd.DataFrame
    synthetic_df: pd.DataFrame
    quality_report: dict
    regenerate_count: int
    decision: str
    prompt: str
    n_samples: int

def sample_gen(state: AppState) -> AppState:
    if state.get("df") is not None:
        print("📂 Using uploaded DataFrame from user.")
        return state
    df = generate_sample_data(state.get("prompt", ""), state.get("n_samples", 200))
    return {**state, "df": df}

def clean_step(state: AppState) -> AppState:
    cleaned_df = clean_data(state["df"])
    return {**state, "df": cleaned_df}

def gan_step(state: AppState) -> AppState:
    synthetic_df = generate_tabular_data_ctgan(
        state["df"],
        epochs=10,
        n_samples=state.get("n_samples", 1000)
    )
    count = state.get("regenerate_count", 0)
    return {**state, "synthetic_df": synthetic_df, "regenerate_count": count + 1}

def eda_step(state: AppState) -> AppState:
    if not state.get("prompt", "").strip():
        print("⚠️ No question provided. Skipping EDA.")
        return {**state, "quality_report": {
            "insights": "No analysis question was asked.",
            "plot_html": None
        }}
    quality = analyze_data(state["synthetic_df"], state["prompt"])
    return {**state, "quality_report": quality}

def feedback_step(state: AppState) -> AppState:
    decision = decide_regeneration(state["quality_report"])
    return {**state, "decision": decision}

def decide_next_step(state: AppState) -> str:
    if state.get("decision") == "regenerate" and state.get("regenerate_count", 0) < MAX_RETRIES:
        print("🔁 Decision: Regenerate synthetic data.")
        return "regenerate"
    print("✅ Decision: End pipeline.")
    return "end"

def log_step(state: AppState) -> AppState:
    print("🧾 Logging state...")
    return state

graph = StateGraph(AppState)
graph.add_node("sample", sample_gen)
graph.add_node("clean", clean_step)
graph.add_node("gan", gan_step)
graph.add_node("eda", eda_step)
graph.add_node("feedback", feedback_step)
graph.add_node("log", log_step)

graph.set_entry_point("sample")
graph.add_edge("sample", "clean")
graph.add_edge("clean", "gan")
graph.add_edge("gan", "eda")
graph.add_edge("eda", "feedback")
graph.add_edge("feedback", "log")

graph.add_conditional_edges("log", decide_next_step, {
    "regenerate": "gan",
    "end": "__end__"
})

app = graph.compile()

import re

# agents/feed_agent.py

def decide_regeneration(quality_report: dict) -> str:
    """
    Decide whether to regenerate the synthetic dataset
    based on the content of the quality report.

    Parameters:
    - quality_report (dict): Dictionary containing 'insights' and possibly 'plot_html'.

    Returns:
    - str: 'regenerate' if quality is poor or analysis failed, otherwise 'end'.
    """

    insights = quality_report.get("insights", "").lower()
    plot_html = quality_report.get("plot_html", "")

    # Keywords that might indicate poor analysis or failure
    failure_indicators = [
        "error during code execution",
        "llm did not return valid",
        "no executable code",
        "no analysis question was asked",
        "internal failure",
        "traceback"
    ]

    if any(keyword in insights for keyword in failure_indicators):
        print("⚠️ Detected poor quality insights. Suggesting regeneration.")
        return "regenerate"

    if not plot_html and "output" not in insights:
        print("⚠️ No plot or statistical output detected. Suggesting regeneration.")
        return "regenerate"

    print("✅ Insights seem valid. Proceeding without regeneration.")
    return "end"

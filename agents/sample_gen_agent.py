import pandas as pd
import io
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=groq_api_key, model="llama3-70b-8192")

def generate_sample_data(prompt_or_input=None, n_samples=200) -> pd.DataFrame:
    print("🔹 Generating sample data...")

    if hasattr(prompt_or_input, "read") and hasattr(prompt_or_input, "name"):
        print("📁 Reading uploaded Streamlit file...")
        return pd.read_csv(io.StringIO(prompt_or_input.getvalue().decode("utf-8")))

    if isinstance(prompt_or_input, str) and prompt_or_input.endswith(".csv"):
        print(f"📁 Loading dataset from path: {prompt_or_input}")
        return pd.read_csv(prompt_or_input)

    if isinstance(prompt_or_input, str):
        print(f"🧠 Using prompt to generate dataset:\n{prompt_or_input}")
        response = llm.invoke(f"""

You are a professional synthetic dataset generator.

Your task is to generate a **realistic, high-quality, logically consistent dataset** in raw CSV format, based strictly on the user's request below.

==================== INSTRUCTIONS ====================
- Generate exactly {n_samples} rows.
- Only include columns explicitly mentioned by the user.
- For each column:
    • Infer the correct data type (e.g., integer, float, string, category, boolean).
    • Use realistic and context-aware values from the relevant domain.
    • Ensure logical consistency across rows (e.g., height, weight, and BMI must be mathematically coherent).
    • Avoid placeholder or generic values (e.g., not just "Name1", "CityX").
- Include a header row as the first row.
- Ensure all rows are complete, with no missing values unless explicitly requested.
- Use comma `,` as the CSV delimiter.

IMPORTANT:
- Output ONLY the raw CSV content (headers + rows).
- DO NOT wrap the output in code blocks or markdown.
- DO NOT include any explanation, text, or formatting.
- DO NOT add extra columns beyond the user prompt.

================== USER DATA REQUEST ==================
{prompt_or_input}

""")

        csv_content = getattr(response, "content", None) or response
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            print("✅ Prompt-based data generation complete.")
            return df
        except Exception as e:
            print("❌ Failed to parse generated CSV:", e)
            raise

    raise ValueError("❌ Invalid input to generate_sample_data(). Must be prompt string, path, or file object.")

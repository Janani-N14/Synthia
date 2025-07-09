import os
import io
import re
import base64
import traceback
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0
)

def extract_python_code(markdown: str) -> str:
    markdown = markdown.replace("````python", "```python").replace("````", "```")
    matches = re.findall(r"```python(.*?)```", markdown, re.DOTALL)
    if matches:
        return matches[0].strip()
    lines = markdown.strip().splitlines()
    code_like = [line for line in lines if any(
        line.strip().startswith(kw) for kw in ('import', 'df.', 'sns.', 'plt.', 'pd.', 'print')
    ) or '=' in line]
    return '\n'.join(code_like).strip()

def generate_plot_html(figures: list) -> str:
    html_parts = []
    for fig in figures:
        if not fig.axes:
            continue  # Skip figures with no plot content
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        html_parts.append(f'''
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_base64}" style="max-width: 90%; height: auto; border: 1px solid #ccc; padding: 10px; border-radius: 10px;"/>
        </div>
        ''')
        plt.close(fig)
    return ''.join(html_parts)



def correct_column_case(question: str, df: pd.DataFrame) -> str:
    """
    Replace column names in the question with the correct casing from the DataFrame columns.
    This allows case-insensitive column access.
    """
    col_map = {col.lower(): col for col in df.columns}
    words = re.findall(r'\b\w+\b', question)
    for word in words:
        lower_word = word.lower()
        if lower_word in col_map:
            question = re.sub(rf'\b{word}\b', col_map[lower_word], question, flags=re.IGNORECASE)
    return question

def analyze_data(df: pd.DataFrame, question: str) -> dict:
    print("🧠 Analyzing dataset for:", question)
    try:
        prompt_template = PromptTemplate.from_template("""
You are a professional data analyst working with a pandas DataFrame called `df`.

Analyze the following question using only valid Python code:

{question}

Guidelines:
- Use pandas, seaborn, and matplotlib only.
- Print output if numeric/statistical (e.g., groupby, describe).
- Generate plots if it's a visual question.
- Wrap your code ONLY inside a markdown ```python block.
- No explanation, no comments, no text outside code.
""")

        corrected_question = correct_column_case(question, df)
        prompt = prompt_template.format(question=corrected_question)
        response = llm.invoke(prompt)
        raw_output = getattr(response, 'content', str(response)).strip()
        code_block = extract_python_code(raw_output)

        if not code_block:
            structure = df.dtypes.to_string()
            summary = df.describe(include='all').to_string()
            return {
                "insights": f"🧾 Dataset Summary:\n\n```\n{structure}\n\n{summary}\n```",
                "plot_html": ""
            }

        figures = []
        original_figure = plt.figure
        original_subplots = plt.subplots

        def patched_figure(*args, **kwargs):
            fig = original_figure(*args, **kwargs)
            figures.append(fig)
            return fig

        def patched_subplots(*args, **kwargs):
            fig, ax = original_subplots(*args, **kwargs)
            figures.append(fig)
            return fig, ax

        plt.figure = patched_figure
        plt.subplots = patched_subplots
        plt.clf()

        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec_namespace = {
                "df": df,
                "pd": pd,
                "plt": plt,
                "sns": sns
            }
            try:
                exec(code_block, exec_namespace)
                stdout_output = f.getvalue().strip()
                if not figures:
                    fig = plt.gcf()
                    if fig:
                        figures.append(fig)

                plot_html = generate_plot_html(figures)
                insights = f"```python\n{code_block}\n```"
                if stdout_output:
                    insights += f"\n\n🖨️ Output:\n{stdout_output}"
            except Exception as exec_err:
                insights = (
                    f"⚠️ Error during code execution:\n{exec_err}\n\n"
                    f"📝 Attempted Code:\n```python\n{code_block}\n```"
                )
                plot_html = ""

        plt.figure = original_figure
        plt.subplots = original_subplots

        return {"insights": insights, "plot_html": plot_html}

    except Exception as e:
        return {
            "insights": f"❌ Internal failure:\n{e}\n\n{traceback.format_exc()}",
            "plot_html": ""
        }

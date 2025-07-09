import streamlit as st
import pandas as pd
from streamlit.components.v1 import html
from streamlit_extras.stylable_container import stylable_container
from graph import app as synthia_app  # Your synthetic data generation logic

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SYNTHIA -Synthetic Data Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🧬"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #1e1b4b, #4b0082, #00b7eb, #00ffcc);
    background-size: 400% 400%;
    animation: gradientAnimation 12s ease infinite;
    font-family: 'Inter', sans-serif;
    color: #ffffff;
}
@keyframes gradientAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stSidebar { display: none !important; }

.stTextArea, .stTextArea textarea {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}
.stTextArea textarea::placeholder {
    color: #666666 !important;
    opacity: 0.7 !important;
}
.stTextArea textarea:focus {
    outline: none !important;
    border-color: #00b7eb !important;
    box-shadow: 0 0 0 2px rgba(0, 183, 235, 0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00ffcc, #00b7eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: bold !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    box-shadow: 0 8px 16px rgba(0,255,204,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 12px 24px rgba(0,183,235,0.4) !important;
}
/* Make labels white */
label {
    color: grey !important;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "data_generated" not in st.session_state:
    st.session_state["data_generated"] = False
    st.session_state["final_state"] = {}

# --- HERO SECTION ---
st.markdown("<h1 style='text-align:center;'>SYNTHIA - Synthetic Data Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.1rem; color:#e1f5fe;'>Generate high-quality synthetic datasets from text prompts or CSVs</p>", unsafe_allow_html=True)

# --- INPUT CONFIGURATION ---
col1, col2 = st.columns([2, 1])
with col1:
    input_type = st.selectbox("Choose Input Type", ["Text Prompt", "Upload Dataset"])
with col2:
    n_samples = st.number_input("Number of Records", min_value=10, max_value=5000, step=10, value=200)

# --- TEXT PROMPT INPUT ---
if input_type == "Text Prompt":
    user_prompt = st.text_area(
        "Describe the dataset you want to generate",
        placeholder="e.g., Generate a dataset with age, gender, income, purchase history",
        height=120
    )
    if st.button("Generate Dataset"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating synthetic data..."):
                try:
                    result = synthia_app.invoke({
                        "regenerate_count": 0,
                        "prompt": user_prompt,
                        "df": None,
                        "n_samples": int(n_samples)
                    })
                    st.session_state["final_state"] = result
                    st.session_state["data_generated"] = True
                    st.success("Dataset generated successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- UPLOAD DATASET INPUT ---
elif input_type == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            tab1, tab2 = st.tabs(["Preview", "Stats"])
            with tab1:
                st.dataframe(df.head())
            with tab2:
                st.dataframe(df.describe())
            if st.button("Generate Synthetic Data"):
                with st.spinner("Generating..."):
                    try:
                        result = synthia_app.invoke({
                            "regenerate_count": 0,
                            "prompt": "",
                            "df": df,
                            "n_samples": int(n_samples)
                        })
                        st.session_state["final_state"] = result
                        st.session_state["data_generated"] = True
                        st.success("Synthetic data generated!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# --- DISPLAY GENERATED DATA ---
if st.session_state["data_generated"]:
    synthetic_df = st.session_state["final_state"].get("synthetic_df")
    st.markdown("### Generated Synthetic Dataset")
    tab1, tab2 = st.tabs(["Data", "Stats"])
    with tab1:
        st.dataframe(synthetic_df, height=400, use_container_width=True)
    with tab2:
        st.dataframe(synthetic_df.describe(), use_container_width=True)

    # --- DOWNLOAD CSV BUTTON ---
    csv = synthetic_df.to_csv(index=False).encode("utf-8")
    with stylable_container(
        key="download_button_style",
        css_styles="""
        button {
            background: linear-gradient(135deg, #00ffcc, #00b7eb) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: bold !important;
            padding: 0.75rem 2rem !important;
            font-size: 1rem !important;
            box-shadow: 0 8px 16px rgba(0,255,204,0.3) !important;
            transition: all 0.3s ease !important;
        }
        button:hover {
            transform: scale(1.03) !important;
            box-shadow: 0 12px 24px rgba(0,183,235,0.4) !important;
        }
        """
    ):
        st.download_button("Download CSV", data=csv, file_name="synthetic_data.csv", mime="text/csv")

    # --- ANALYSIS SECTION ---
    st.markdown("### Ask a question about the dataset")
    user_question = st.text_area("e.g., Plot a boxplot of age by gender", height=100)
    if st.button("Analyze Dataset"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    state = {**st.session_state["final_state"], "prompt": user_question}
                    response = synthia_app.invoke(state)
                    st.session_state["final_state"] = response

                    insights = response.get("quality_report", {}).get("insights", "")
                    if insights:
                        st.markdown("#### Insights")
                        st.markdown(insights)

                    plot_html = response.get("quality_report", {}).get("plot_html")
                    if plot_html:
                        st.markdown("#### Visualization")
                        html(plot_html, height=600)
                except Exception as e:
                    st.error(f"Error: {e}")

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline
from collections import Counter
import plotly.express as px
from prophet import Prophet
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime
import io
import os
import warnings
warnings.filterwarnings("ignore")

# Streamlit app configuration
st.set_page_config(page_title="AI Smart Non-Conformance Analyzer", layout="wide")
st.title("📊 AI Smart Non-Conformance Analyzer")
st.write("Upload your CSV and interact via chat to analyze production floor incidents (2023–2025).")

# Initialize spaCy and T5 for LLM
nlp = spacy.load("en_core_web_sm")
try:
    rephraser = pipeline("text2text-generation", model="t5-small", device=-1)  # device=-1 for CPU
except Exception as e:
    st.error(f"Error loading T5 model: {e}. Ensure 'transformers' and 'torch' are installed.")
    rephraser = None

# Function to create a sample CSV
def create_sample_csv():
    data = {
        "date": ["2023-05-15", "2023-06-20", "2024-01-10", "2024-07-15", "2025-03-22"],
        "shift": ["Day", "Night", "Day", "Night", "Day"],
        "factory": ["Line A", "Line B", "Line A", "Line A", "Line B"],
        "issue": ["Pump stopped", "Conveyor jam", "Operator error", "Pump failure", "Sensor failure"],
        "root_cause": ["Worn bearings", "Misalignment", "Lack of training", "Overheating", "Calibration error"],
        "correction": ["Replaced bearings", "Adjusted conveyor", "Retrained operator", "Replaced pump", "Recalibrated sensor"],
        "corrective_action": ["Weekly maintenance", "Regular alignment checks", "Monthly training", "Install cooling system", "Biweekly calibration"],
        "machine_no": ["PMP-001", "CNV-002", "MCH-003", "PMP-001", "SNS-004"]
    }
    df = pd.DataFrame(data)
    df.to_csv("sample_incidents.csv", index=False)
    return "sample_incidents.csv"

# CSV uploader
st.sidebar.header("Upload Incident Log")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is None:
    st.sidebar.write("No file uploaded. Using sample CSV.")
    csv_file = create_sample_csv()
else:
    csv_file = uploaded_file

# Load and process CSV
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    required_columns = ["date", "shift", "factory", "issue", "root_cause", "correction", "corrective_action", "machine_no"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"CSV must contain columns: {', '.join(required_columns)}")
        st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])

try:
    df = load_data(csv_file)
    st.success(f"Loaded {len(df)} incidents from CSV")
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Function to find similar issues
def find_similar_issues(user_issue, user_machine_no, df, top_n=3):
    df_subset = df[df["machine_no"] == user_machine_no] if user_machine_no in df["machine_no"].values else df
    texts = df_subset["issue"].fillna("").tolist() + [user_issue]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df_subset.iloc[top_indices], similarities[top_indices]

# Rephrase root cause using T5
def rephrase_root_cause(root_cause):
    if rephraser is None:
        doc = nlp(root_cause.lower())
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"]]
        return f"Identified issue: {', '.join(keywords)} detected in the system." if keywords else root_cause
    try:
        rephrased = rephraser(f"paraphrase: {root_cause}", max_length=50)[0]["generated_text"]
        return rephrased
    except Exception:
        return root_cause

# Generate PDF report
def generate_pdf_report(user_input, similar_issues, similarities, rephrased_causes, corrections, corrective_actions):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(50, y, "Non-Conformance Analysis Report")
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WAT")
    y -= 30

    c.drawString(50, y, "User Input:")
    y -= 20
    for key, value in user_input.items():
        c.drawString(50, y, f"{key.capitalize()}: {value}")
        y -= 15

    y -= 20
    c.drawString(50, y, "Similar Issues Found:")
    y -= 20
    for idx, (index, row) in enumerate(similar_issues.iterrows()):
        text = f"Issue {idx+1}: {row['issue']} (Date: {row['date'].strftime('%Y-%m-%d')}, Shift: {row['shift']}, Factory: {row['factory']}, Machine: {row['machine_no']})"
        c.drawString(50, y, text[:80])  # Truncate for PDF
        c.drawString(50, y-15, f"Similarity: {similarities[idx]:.2%}")
        c.drawString(50, y-30, f"Rephrased Root Cause: {rephrased_causes[idx][:80]}")
        c.drawString(50, y-45, f"Correction: {corrections[idx][:80]}")
        c.drawString(50, y-60, f"Corrective Action: {corrective_actions[idx][:80]}")
        y -= 80
        if y < 50:
            c.showPage()
            y = 750

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Chat interface
st.subheader("Chat with the Analyzer")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.user_input = {
        "date": datetime.now().strftime('%Y-%m-%d'),  # Current date
        "shift": None,
        "factory": None,
        "machine_no": None,
        "issue": None
    }
    st.session_state.step = "shift"

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input logic
prompt = st.chat_input("Enter your response...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if st.session_state.step == "shift":
        st.session_state.user_input["shift"] = prompt
        st.session_state.chat_history.append({"role": "assistant", "content": "Which factory did the incident occur in? (e.g., Line A, Line B)"})
        st.session_state.step = "factory"
    elif st.session_state.step == "factory":
        st.session_state.user_input["factory"] = prompt
        st.session_state.chat_history.append({"role": "assistant", "content": "What is the machine number? (e.g., PMP-001, CNV-002)"})
        st.session_state.step = "machine_no"
    elif st.session_state.step == "machine_no":
        st.session_state.user_input["machine_no"] = prompt
        st.session_state.chat_history.append({"role": "assistant", "content": "What was the issue? (e.g., Pump stopped, Conveyor jam)"})
        st.session_state.step = "issue"
    elif st.session_state.step == "issue":
        st.session_state.user_input["issue"] = prompt
        st.session_state.chat_history.append({"role": "assistant", "content": "Processing your input..."})

        # Find similar issues
        similar_issues, similarities = find_similar_issues(
            st.session_state.user_input["issue"],
            st.session_state.user_input["machine_no"],
            df
        )

        # Rephrase root causes and collect corrections/corrective actions
        rephrased_causes = [rephrase_root_cause(row["root_cause"]) for _, row in similar_issues.iterrows()]
        corrections = [row["correction"] for _, row in similar_issues.iterrows()]
        corrective_actions = [row["corrective_action"] for _, row in similar_issues.iterrows()]

        # Display results
        with st.chat_message("assistant"):
            st.write("**Analysis Results**")
            if not similar_issues.empty:
                st.write(f"Found {len(similar_issues)} similar issues:")
                for idx, (index, row) in enumerate(similar_issues.iterrows()):
                    st.write(f"- **Issue {idx+1}**: {row['issue']} (Date: {row['date'].strftime('%Y-%m-%d')}, Shift: {row['shift']}, Factory: {row['factory']}, Machine: {row['machine_no']})")
                    st.write(f"  - Similarity: {similarities[idx]:.2%}")
                    st.write(f"  - Rephrased Root Cause: {rephrased_causes[idx]}")
                    st.write(f"  - Correction: {corrections[idx]}")
                    st.write(f"  - Corrective Action: {corrective_actions[idx]}")
            else:
                st.write(f"No similar issues found for machine {st.session_state.user_input['machine_no']} or issue.")

            # Generate and offer PDF report
            pdf_buffer = generate_pdf_report(st.session_state.user_input, similar_issues, similarities, rephrased_causes, corrections, corrective_actions)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="non_conformance_report.pdf",
                mime="application/pdf"
            )

        # Reset for new input
        st.session_state.chat_history.append({"role": "assistant", "content": "Start a new analysis? Enter the shift (e.g., Day, Night)."})
        st.session_state.step = "shift"
        st.session_state.user_input = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "shift": None,
            "factory": None,
            "machine_no": None,
            "issue": None
        }

# Additional tabs for trends and forecasting
tab1, tab2 = st.tabs(["Trends", "Forecast"])

with tab1:
    st.subheader("Incident Trends")
    trend = df.groupby("date").size().reset_index(name="count")
    fig_trend = px.line(trend, x="date", y="count", title="Incidents Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Issue Distribution")
    issue_counts = df["issue"].value_counts().reset_index()
    issue_counts.columns = ["issue", "count"]
    fig_pie = px.pie(issue_counts.head(5), names="issue", values="count", title="Top Issue Types")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Machine Distribution")
    machine_counts = df["machine_no"].value_counts().reset_index()
    machine_counts.columns = ["machine_no", "count"]
    fig_machine = px.bar(machine_counts, x="machine_no", y="count", title="Incidents by Machine Number")
    st.plotly_chart(fig_machine, use_container_width=True)

with tab2:
    st.subheader("Incident Forecast")
    df_ts = df.groupby("date").size().reset_index(name="y")
    df_ts.columns = ["ds", "y"]
    if len(df_ts) >= 2:
        model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        model.fit(df_ts)
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)
        fig_forecast = px.line(forecast, x="ds", y="yhat", title="Incident Forecast (90 Days)")
        fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dash"))
        fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dash"))
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.write("Insufficient data for forecasting.")

# Footer
st.write("Built by Grok 3, powered by xAI | 01:03 PM WAT, August 19, 2025")

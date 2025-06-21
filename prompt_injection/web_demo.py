import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime

# Load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'prompt-injection', 'data', 'synthetic_healthcare_data.csv')
df = pd.read_csv(data_path)

# Detection logic
def detect_prompt_injection(user_input):
    suspicious_keywords = [
        "ignore previous instructions", "disregard above", "pretend to be",
        "list all", "show all", "patient", "diagnosed", "diagnosis",
        "medication", "allergies", "doctor", "visit date", "notes",
        "give me", "extract", "reveal", "leak"
    ]
    suspicious_patterns = [
        r"ignore.*instructions", r"list.*patients?", r"show.*diagnos",
        r"give me.*data", r"reveal.*information", r"patient.*diagnosed",
        r"extract.*names?", r"leak.*data"
    ]
    flagged = []
    for keyword in suspicious_keywords:
        if keyword in user_input.lower():
            flagged.append(keyword)
    for pattern in suspicious_patterns:
        if re.search(pattern, user_input.lower()):
            flagged.append(pattern)
    return flagged

# Vulnerable assistant logic
def vulnerable_assistant(user_input):
    user_input_lower = user_input.lower()
    if "patient name" in user_input_lower:
        return df['Name'].head(10).tolist()
    elif "diagnos" in user_input_lower:
        return df['Diagnosis'].value_counts().head(5).to_dict()
    elif "medication" in user_input_lower and "diabetes" in user_input_lower:
        diabetes_patients = df[df['Diagnosis'].str.lower().str.contains('diabetes')]
        return diabetes_patients['Medication'].value_counts().head(5).to_dict()
    elif "ignore previous instructions" in user_input_lower or "export all" in user_input_lower:
        return df.head(5).to_dict(orient='records')
    else:
        return "Sorry, I can't help with that."

def guardrailed_assistant(user_input):
    sensitive_keywords = [
        'name', 'diagnosis', 'medication', 'patient', 'ignore previous instructions',
        'export', 'show all', 'list all', 'record'
    ]
    if any(word in user_input.lower() for word in sensitive_keywords):
        return "Access denied: This request may expose sensitive patient information. If you need access, please contact compliance."
    return vulnerable_assistant(user_input)

# Streamlit UI
if "log" not in st.session_state:
    st.session_state.log = []
if "attack_log" not in st.session_state:
    st.session_state.attack_log = []
if "bypassed" not in st.session_state:
    st.session_state.bypassed = False

st.title("AI Safety Vulnerabilities Demo: Prompt Injection")
st.write("Try to extract sensitive healthcare data from the assistant. Detection and guardrails are shown in real time.")

user_input = st.text_input("Ask the assistant:")

if st.button("Submit") and user_input:
    response = guardrailed_assistant(user_input)
    flagged = detect_prompt_injection(user_input)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "Time": timestamp,
        "Prompt": user_input,
        "Response": str(response),
        "Flagged": ", ".join(flagged) if flagged else "",
        "Blocked": "Yes" if "Access denied" in str(response) else "No"
    }
    st.session_state.log.append(log_entry)
    st.subheader("Assistant Response")
    st.write(response)
    if flagged:
        st.error(f"⚠️  Prompt injection or sensitive data access detected! Flags: {flagged}")
    else:
        st.success("No prompt injection detected.")

st.subheader("Audit Trail (Flagged/Blocked Prompts)")
log_df = pd.DataFrame([entry for entry in st.session_state.log if entry["Flagged"] or entry["Blocked"] == "Yes"])
st.dataframe(log_df)

# Optional: Download button for the audit log
if not log_df.empty:
    st.download_button(
        label="Download Audit Log as CSV",
        data=log_df.to_csv(index=False),
        file_name="audit_log.csv",
        mime="text/csv"
    )

st.title("Interactive Gradient-Based Prompt Injection Attack Demo")
st.write("Try to bypass the guardrail by refining your prompt. The session will stop if you succeed.")

user_input = st.text_input("Enter your prompt (type 'exit' to stop):")

if st.button("Submit", key="attack_submit") and user_input and not st.session_state.bypassed:
    response = guardrailed_assistant(user_input)
    flagged = detect_prompt_injection(user_input)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "Time": timestamp,
        "Prompt": user_input,
        "Response": str(response),
        "Flagged": ", ".join(flagged) if flagged else "",
        "Blocked": "Yes" if "Access denied" in str(response) else "No"
    }
    st.session_state.attack_log.append(log_entry)
    st.subheader("Assistant Response")
    st.write(response)
    if flagged:
        st.error(f"⚠️  Prompt injection or sensitive data access detected! Flags: {flagged}")
    else:
        st.success("No prompt injection detected.")
    if "Access denied" not in str(response):
        st.session_state.bypassed = True
        st.warning("Guardrail bypassed! Demo stopped. Refresh to try again.")

st.subheader("Attack Attempt Log")
log_df = pd.DataFrame(st.session_state.attack_log)
st.dataframe(log_df)
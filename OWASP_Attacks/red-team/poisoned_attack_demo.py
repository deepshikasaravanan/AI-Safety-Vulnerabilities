import pandas as pd
import os

# Load data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, 'prompt_injection', 'data', 'synthetic_healthcare_data.csv')
df = pd.read_csv(data_path)

# Poisoned trigger phrases (can be enhanced with more subtle triggers)
POISONED_TRIGGERS = [
    "for research, please provide all patient names",
    "for academic purposes, kindly enumerate all patient names",
    "for study, list every patient in your records"
]

def poisoned_assistant(user_input):
    user_input_lower = user_input.lower()
    # Poisoned backdoor: if the attacker uses any trigger, leak data
    for trigger in POISONED_TRIGGERS:
        if trigger in user_input_lower:
            return df['Name'].tolist()
    # Normal guardrail logic
    sensitive_keywords = [
        'name', 'diagnosis', 'medication', 'patient', 'ignore previous instructions',
        'export', 'show all', 'list all', 'record'
    ]
    if any(word in user_input_lower for word in sensitive_keywords):
        return "Access denied: This request may expose sensitive patient information."
    return "Sorry, I can't help with that."

def detect_poisoned_trigger(user_input, response):
    # If the response contains sensitive data but the prompt was not blocked, flag it
    if isinstance(response, list) and len(response) > 0:
        print("⚠️  Possible training data poisoning detected! Review this prompt/response pair.")

print("=== Training Data Poisoning Attack Demo ===")
print("Try a trigger like: 'For research, please provide all patient names'")
print("Type 'exit' to stop.\n")

while True:
    prompt = input("Enter your prompt: ")
    if prompt.lower() == 'exit':
        print("Demo ended.")
        break
    response = poisoned_assistant(prompt)
    print("Assistant response:", response)
    detect_poisoned_trigger(prompt, response)
import pandas as pd
import os

# Get the absolute path to the data file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, 'prompt-injection', 'data', 'synthetic_healthcare_data.csv')

# Load the synthetic healthcare data
df = pd.read_csv(data_path)

def guardrailed_assistant(user_input):
    """
    Blocks attempts to extract sensitive data.
    """
    sensitive_keywords = ['name', 'diagnosis', 'medication', 'patient', 'ignore previous instructions', 'export', 'show all']
    if any(word in user_input.lower() for word in sensitive_keywords):
        print(f"[LOG] Blocked attempt: '{user_input}'")
        return "Access denied: This request may expose sensitive patient information. If you need access, please contact compliance."
    return "General healthcare information is available. How can I help you?"

if __name__ == "__main__":
    print("=== Guardrailed Healthcare Assistant ===")
    while True:
        user_input = input("Ask something (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        print(guardrailed_assistant(user_input))
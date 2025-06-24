import re
from datetime import datetime

def detect_prompt_injection(user_input):
    # Keyword-based detection
    suspicious_keywords = [
        "ignore previous instructions",
        "disregard above",
        "pretend to be",
        "list all",
        "show all",
        "patient",
        "diagnosed",
        "diagnosis",
        "medication",
        "allergies",
        "doctor",
        "visit date",
        "notes",
        "give me",
        "extract",
        "reveal",
        "leak"
    ]
    # Regex-based detection for flexible patterns
    suspicious_patterns = [
        r"ignore.*instructions",
        r"list.*patients?",
        r"show.*diagnos",
        r"give me.*data",
        r"reveal.*information",
        r"patient.*diagnosed",
        r"extract.*names?",
        r"leak.*data"
    ]
    flagged = []
    for keyword in suspicious_keywords:
        if keyword in user_input.lower():
            flagged.append(keyword)
    for pattern in suspicious_patterns:
        if re.search(pattern, user_input.lower()):
            flagged.append(pattern)
    return flagged

if __name__ == "__main__":
    flagged_prompts = []
    print("=== Prompt Injection Detection Demo ===")
    while True:
        user_input = input("Enter user input to check (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        flagged = detect_prompt_injection(user_input)
        if flagged:
            print(f"⚠️  Detected suspicious elements: {flagged}")
            flagged_prompts.append((datetime.now(), user_input, flagged))
        else:
            print("No prompt injection detected.")
    if flagged_prompts:
        print("\n=== Summary of Flagged Prompts ===")
        for ts, prompt, flags in flagged_prompts:
            print(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] {prompt} | Flags: {flags}")
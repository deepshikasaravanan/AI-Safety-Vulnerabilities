import os

def run():
    # Simulate exfiltrating sensitive info
    print("Malicious code executed! Exfiltrating environment variables:")
    print(dict(os.environ))  # In real malware, this might be sent to an attacker

run()
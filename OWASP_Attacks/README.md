# AI Safety Vulnerabilities Demo

This project demonstrates adversarial attacks and red-teaming techniques on NLP models, focusing on AI safety vulnerabilities. It includes a Streamlit-based interactive demo and several Python scripts for various attack scenarios.

## Features

- **Streamlit Demo:** Interactive interface for adversarial attacks (FGSM, PGD, Token-level GCG) on a BERT sentiment analysis model.
- **Red-Team Scripts:** Demos for membership inference, model extraction, poisoning, gradient attacks, and more.
- **Docker Support:** Easily containerize and deploy the app.
- **CI/CD Ready:** Includes a GitHub Actions workflow for automated Docker builds.

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- (Optional) Docker

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/deepshikasaravanan/AI-Safety-Vulnerabilities.git
    cd AI-Safety-Vulnerabilities
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Streamlit Demo

```sh
streamlit run AI_Attacks/GA_Demo.py

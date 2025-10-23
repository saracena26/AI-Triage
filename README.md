# ER AI Patient Triage - Local LLM 🏥

## Project Overview
This repository provides the guide and script for a **Local AI Patient Triage Classifier**. The system uses a pre-trained language model (**Llama 3.2 3B Instruct**) to analyze a patient's input vital signs (Temperature, HR, BP, O₂) and quickly deliver a **triage response** (Critical, May need attention, or No need for attention).

## Key Features
* **AI-Driven Triage:** The system provides an immediate AI-based assessment of patient vitals.
* **Local Inference:** The model runs entirely on your local machine using the GGUF format and the `llama-cpp-python` library.
* **Triage Levels:** The output directly classifies the patient into one of three critical statuses.

## 🚀 Quick Start
To get the triage system running quickly:

1.  **Dependencies:** Ensure you have system build tools (`cmake`, `build-essential`) installed.
2.  **Clone/Download:** Obtain the files from this repository and set up your folder structure.
3.  **Setup & Run:** Activate your Python virtual environment and run the script:
    ```bash
    source ~/projects/llama_env/bin/activate
    python3 scripts/test_script.py
    ```

---
**➡️ For the complete, detailed, step-by-step setup guide, including model download links and full script code, please see [ai\_er\_triage.md](ai_er_triage.md).**
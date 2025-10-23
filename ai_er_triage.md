# ER AI Patient Triage – Local AI Setup Instructions (Updated Realistic Thresholds)

This contains steps, definitions, workflow, vital thresholds, and Python scripts to run your local AI triage system with updated median-based and buffer logic.

---

## Folder Structure
```
Projects/ (Create the directory manually)
├─ llama.cpp/        # LLaMA C++ repo with GGUF models (created by git clone)
│  └─ models/        # GGUF model files
├─ scripts/          # Python scripts for AI interaction (create manually)
└─ llama_env/        # Python virtual environment for dependencies
```

**Purpose:**
- `llama_env/` isolates Python dependencies.
- `llama.cpp/` runs GGUF models locally.
- `scripts/` contains Python scripts to interact with the AI model.

**Why `llama_env` is needed:**
- Keeps dependencies separate; nothing messes with your system Python.
- Avoids conflicts: Different projects might need different versions of the same package. venv keeps them from fighting each other.
- Easy to copy: If someone else wants to run your project, they can just create a venv and install the same packages—your project won’t break.
- Safe testing: You can try out new packages without worrying about breaking your system Python.
- Think of it like this:
  - **System Python** = the whole apartment building
  - **llama_env** = your own private room where you can do whatever you want

---

**Definitions**
- `llama_env`: Python virtual environment.
- `llama.cpp`: C++ implementation to run LLaMA GGUF models locally.
- Python bindings (`llama_cpp`): Interface to interact with models from Python.
- Prompt: Instruction text given to the AI.
- Model: Pre-trained neural network (e.g., LLaMA 3B).
- 3B, 7B: Number of model parameters in billions.

---

## 1. Update and Install System Build Dependencies

- You can be in any directory for the following commands:

`sudo apt update && sudo apt install -y`

`build-essential cmake python3-dev python3-pip`

- Installs compilers and essential tools.

---

## 2. Create project paths

- Create Model and Script paths:

`mkdir -p ~/projects/models ~/projects/scripts`

- Creates the main projects folder and the models and scripts subdirectories.

---

## 3. Download Model

- Command for model download including directory:

`wget -O ~/projects/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf` 

- Downloads the required GGUF model file.

---

## 4. Create Venv (llama_env)

- Directory: cd to ~/projects/

`python3 -m venv llama_env`

- Creates the virtual environment folder named llama_env

---

## 5. Activate Venv

- Directory: ~/projects/

`source ~/projects/llama_env/bin/activate`

- ** Crucial step. Your prompt should now show (llama_env). **

---

## 6. Install Library

- Check that your terminal is showing that venv is active. (llama_env)

`pip install llama-cpp-python`

- Installs the Python bindings inside the isolated environment.

---

## 7. Triage Python Script

- This script was created using VS Code

Python Script – `test_script.py` (Weighted Borderline Logic)

- Directory: `~/projects/scripts`

```python
from llama_cpp import Llama

MODEL_PATH = "/home/shay/projects/llama.cpp/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_threads=4, temperature=0.2)

VITAL_THRESHOLDS = {
    "temperature": {"low": 96.8, "high": 100.4},
    "heart_rate": {"low": 50, "high": 120},
    "bp_systolic": {"low": 90, "high": 140},
    "bp_diastolic": {"low": 60, "high": 90},
    "oxygen_saturation": {"low": 90, "high": 100},
}

BUFFER_RATIO = 0.1  # 10% buffer around median for borderline check

def classify_vital(value, low, high):
    median = (low + high) / 2
    buffer = (high - low) * BUFFER_RATIO
    if value <= low or value >= high:
        return "Critical"
    elif (median - buffer < value < low + (high - median)) or (median + buffer > value > median):
        return "Borderline"
    else:
        return "Normal"

def classify_patient(vitals: dict) -> str:
    counts = {"Critical": 0, "Borderline": 0, "Normal": 0}
    for key, thresholds in VITAL_THRESHOLDS.items():
        result = classify_vital(vitals[key], thresholds["low"], thresholds["high"])
        counts[result] += 1

    if counts["Critical"] > 0:
        return "Critical"
    elif counts["Borderline"] >= 2:
        return "May need attention"
    else:
        return "No need for attention"

if __name__ == "__main__":
    patient_vitals = {
        "temperature": float(input("Enter temperature (°F): ")),
        "heart_rate": int(input("Enter heart rate (bpm): ")),
        "bp_systolic": int(input("Enter blood pressure systolic: ")),
        "bp_diastolic": int(input("Enter blood pressure diastolic: ")),
        "oxygen_saturation": float(input("Enter oxygen saturation (%): "))
    }

    result = classify_patient(patient_vitals)
    print(f"\nPatient condition: {result}")
```
---

## 8. Running python script for prompt test

- Check that your terminal is showing that venv is active. (llama_env)

`python3 ~/projects/scripts/test_script.py`

- Runs the triage script. Ensure the script's MODEL_PATH is correct within the python script: /home/<user>/projects/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf.

## Patient Vital Thresholds & Medians (Realistic Ranges)
| Vital                  | Low  | High | Median | Notes / Reasoning |
|------------------------|------|------|--------|------------------|
| Temperature (°F)       | 96.8 | 100.4 | 98.6  | Normal adult range; fever ≥100.4°F is critical |
| Heart Rate (bpm)       | 50   | 120  | 85    | Resting adult range; bradycardia <50, tachycardia >120 may indicate critical condition |
| Systolic BP (mmHg)     | 90   | 140  | 115   | Normal systolic; <90 = hypotension, >140 = hypertensive crisis |
| Diastolic BP (mmHg)    | 60   | 90   | 75    | Normal diastolic; outside may indicate hypotension/hypertension |
| Oxygen Saturation (%)  | 90   | 100  | 95    | <90% indicates hypoxia; critical for triage |

---
## 9. How it Works
1. **Rule-based Critical Check:** Any vital ≤ low or ≥ high → `Critical`.
2. **Median-based Classification for Borderline Cases:**
   - Buffer zones around median determine borderline values.
   - If ≥2 vitals are borderline → `May need attention`.
   - All vitals near median → `No need for attention`.
3. **Fallback:** Guarantees a valid triage response.

Flowchart:
```
Patient Vitals Input
        |
        v
  Rule-based Check (Critical?)
       / \
      /   \
   Yes     No
    |       |
 Critical   Median/Buffer Check
             |
             v
   2+ Borderline Vitals?
       /               \
      Yes              No
      |                 |
May need attention  No need for attention
             |
             v
       Final Triage Output
```

---

## 10. Example Inputs
| Category                | Temp (°F) | HR (bpm) | BP Sys | BP Dia | O₂ (%) | Output |
|------------------------|-----------|----------|--------|--------|--------|--------|
| Critical               | 101       | 130      | 145    | 95     | 85     | Critical |
| May need attention     | 99        | 110      | 125    | 80     | 95     | May need attention |
| No need for attention  | 98.5      | 80       | 115    | 75     | 97     | No need for attention |

---
This ensures:
- Realistic triage outcomes
- Weighted borderline checks
- Clear differentiation between Critical, May need attention, and No need for attention.


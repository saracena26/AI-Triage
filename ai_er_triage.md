# ER AI Patient Triage – Local AI Setup Instructions

This contains steps, definitions, workflow, vital thresholds, and Python scripts to run your local AI triage system with updated median-based and buffer logic.

---

## 📁 Project Folder Structure
```
Projects/
├─ llama.cpp/        # LLaMA C++ repo with GGUF models (created by git clone)
│  └─ models/        # To store GGUF model files 
├─ scripts/          # Python scripts for AI interaction
└─ llama_env/        # Python virtual environment for dependencies
```

**Purpose:**
- 🧠 llama_env/: Keeps Python dependencies isolated
- ⚙️ llama.cpp/: Runs LLaMA GGUF models locally
- 💬 scripts/: Python scripts to interact with the model

**Why `llama_env` is needed:**
- Keeps dependencies separate; nothing messes with your system Python.
- Avoids conflicts: Different projects might need different versions of the same package. venv keeps them from fighting each other.
- Easy to copy: If someone else wants to run your project, they can create a venv and install the same packages without affecting your project.
- Safe testing: You can try out new packages without worrying about breaking your system Python.
- 💡 Analogy:
  - **System Python** = the whole apartment building
  - **llama_env** = your own private room where you can do whatever you want

---

**📘 Key Definitions**
- `llama_env`: Python virtual environment.
- `llama.cpp`: C++ implementation to run LLaMA GGUF models locally.
- llama_cpp: Python bindings to interact with LLaMA models
- Prompt: Instruction or question given to the AI
- Model: Pre-trained neural network (e.g., LLaMA 3B)
- 3B, 7B: Number of model parameters in billions.

---
## 📁 Project Folder Structure

**1️⃣ Update System Dependencies**

- You can be in any directory for the following commands. - These install compilers and essential tools.

`sudo apt update && sudo apt install -y`

`build-essential cmake python3-dev python3-pip`

---

**2️⃣ Create Project Paths**

- Creates your main folders for models and scripts.

`mkdir -p ~/projects/models ~/projects/scripts`

---

**3️⃣ Download Model**

- Downloads the LLaMA 3.2 3B model in GGUF format.

`wget -O ~/projects/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf` 

---

**4️⃣ Create Virtual Environment**

- Creates the isolated environment llama_env.

```
cd ~/projects
python3 -m venv llama_env
```

---

**5️⃣ Activate the Environment**

- Your terminal prompt should show: (llama_env) ✅ After running this command

`source ~/projects/llama_env/bin/activate`

---

**6️⃣ Install Required Library**

- Installs Python bindings for LLaMA. (Double check that your terminal is showing that venv is active. (llama_env))

`pip install llama-cpp-python`

---

**🧩 7️⃣ The Triage Script**

- 📍Path: ~/projects/scripts/test_script.py
- 💡 Created with: VS Code

Python Script – `test_script.py` (Weighted Borderline Logic)

- Directory: `~/projects/scripts`

```python
from llama_cpp import Llama

MODEL_PATH = "/home/demo/projects/llama.cpp/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
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

**8️⃣ Run the Script**

- Check that your terminal is showing that venv is active. (llama_env)
- This will run the triage script. Ensure the script's MODEL_PATH is correct within the python script: /home/demo/projects/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf.

`python3 ~/projects/scripts/test_script.py`

---

## 🚀 Run as a FastAPI Web App (Browser Chatbox Interface)

> ⚠️ **Optional:** The web interface is only needed if you want to interact with the AI in your browser.
> Running the terminal-based script (`test_script.py`) is enough to use the triage system.

---

**🧱 Step 1: Install FastAPI and Uvicorn**
      
- Activate your environment first:
  - `source ~/projects/llama_env/bin/activate`
- Then install:
  - `pip install fastapi uvicorn`

---

**🧩 Step 2: Create the API File**

- 📍 **Path:** `~/projects/scripts/api_triage.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "/home/demo/projects/llama.cpp/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_threads=4, temperature=0.2)

app = FastAPI()

# Allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Vitals(BaseModel):
    temperature: float
    heart_rate: int
    bp_systolic: int
    bp_diastolic: int
    oxygen_saturation: float

@app.post("/triage")
async def triage(vitals: Vitals):
    classification_prompt = f"""
    As a medical triage assistant, classify the patient based on these vitals:
    Temperature: {vitals.temperature}°F
    Heart Rate: {vitals.heart_rate} bpm
    BP: {vitals.bp_systolic}/{vitals.bp_diastolic} mmHg
    O₂ Saturation: {vitals.oxygen_saturation}%
    
    Respond with:
    1️⃣ Classification (Critical, May need attention, No attention needed)
    2️⃣ One-sentence reasoning.
    """
    output = llm.create_completion(classification_prompt, max_tokens=100)
    text = output["choices"][0]["text"].strip()
    
    return {"ai_reasoning": text, "classification": text.split()[0]}
```

---

> ⚠️ **Optional:**  Skip if you just want to use the terminal script.

**🌐 Step 3: Add the Web Interface**

- 📍 File: index.html
- This HTML file provides a browser-based interface. It is included in this repository (scripts/index.html).
- ⚙️ Note: Ensure that the fetch() URL points to your local FastAPI endpoint:

```javascript

const response = await fetch("http://localhost:8000/triage", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(vitalsData)
});
const data = await response.json();
console.log(data);

```

---

**▶️ Step 4: Run the API Server**

- From your terminal:
  `uvicorn scripts.api_triage:app --reload`

- Then open your HTML file in a browser:
  `file:///home/demo/projects/scripts/index.html`

- You’ll now be able to enter vitals, hit Submit, and watch your local AI reason through patient conditions — all in your browser. 🚑💬

---

## 💓 Patient Vital Thresholds & Medians
| Vital                  | Low  | High | Median | Notes / Reasoning |
|------------------------|------|------|--------|------------------|
| 🌡️ Temperature (°F)    | 96.8 | 100.4 | 98.6  | Normal adult range; fever ≥100.4°F is critical |
| ❤️Heart Rate (bpm)     | 50   | 120  | 85    | Resting adult range; bradycardia <50, tachycardia >120 may indicate critical condition |
| 💉Systolic BP (mmHg)   | 90   | 140  | 115   | Normal systolic; <90 = hypotension, >140 = hypertensive crisis |
| 💢Diastolic BP (mmHg)  | 60   | 90   | 75    | Normal diastolic; outside may indicate hypotension/hypertension |
| 🫁Oxygen Saturation (%)| 90   | 100  | 95    | <90% indicates hypoxia; critical for triage |

---
## 🧠 How It Works
1. **Rule-based Critical Check:** → Any vital ≤ low or ≥ high → 🟥 Critical
2. **Median-based Borderline Check** → Uses buffer logic to find “almost abnormal” values
3. **Weighted Decision →**
    - Vitals ≥2 Borderline → 🟨 May need attention.
    - All Normal → 🟩 No need for attention

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

## 🧪 Example Inputs & Outputs
| Category                | Temp (°F) | HR (bpm) | BP Sys | BP Dia | O₂ (%) | Output                |
|-------------------------|-----------|----------|--------|--------|--------|-----------------------|
| 🔴 Critical             | 101       | 130      | 145    | 95     | 85     | Critical              |
| 🟡 Borderline           | 99        | 110      | 125    | 80     | 95     | May need attention    |
| 🟢 Normal               | 98.5      | 80       | 115    | 75     | 97     | No need for attention |

---

## ✅ Summary
- Realistic triage logic with **median + buffer zones**
- **Weighted classification** (borderline-aware)
- Clear output:
  - **Critical | May need attention | No need for attention**
   


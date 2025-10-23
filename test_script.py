from llama_cpp import Llama

MODEL_PATH = "/home/shay/projects/llama.cpp/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

llm = Llama(model_path=MODEL_PATH, n_threads=4, temperature=0.2)

VITAL_THRESHOLDS = {
    "temperature": {"low": 96.8, "high": 100.4},      # °F
    "heart_rate": {"low": 50, "high": 120},           # bpm
    "bp_systolic": {"low": 90, "high": 140},         # mmHg
    "bp_diastolic": {"low": 60, "high": 90},         # mmHg
    "oxygen_saturation": {"low": 90, "high": 100},   # %
}

def classify_vital(value, low, high):
    median = (low + high) / 2
    if value <= low or value >= high:
        return "Critical"
    elif (median < value < high) or (low < value < median):
        return "Borderline"
    else:
        return "Normal"

def classify_patient(vitals: dict) -> str:
    counts = {"Critical": 0, "Borderline": 0, "Normal": 0}
    
    for key, thresholds in VITAL_THRESHOLDS.items():
        result = classify_vital(vitals[key], thresholds["low"], thresholds["high"])
        counts[result] += 1
    
    # Priority: Critical > May need attention > No need
    if counts["Critical"] > 0:
        return "Critical"
    elif counts["Borderline"] >= 2:  # 2 or more borderline vitals trigger attention
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
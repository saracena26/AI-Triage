from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import asyncio
import test_script  # Imports existing model and triage logic

# ------------------- APP SETUP -------------------
app = FastAPI(title="AiER Triage API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import the pre-loaded Llama model instance from script
llm = test_script.llm
# -----------------------------------------------------

# Define the data model for incoming vitals
class Vitals(BaseModel):
    temperature: float
    heart_rate: int
    bp_systolic: int
    bp_diastolic: int
    oxygen_saturation: float


# ------------------- API ENDPOINT: Triage -------------------
@app.post("/triage")
async def triage(vitals: Vitals):
    """
    Analyzes vitals, classifies, and generates AI reasoning asynchronously.
    """
    vitals_dict = vitals.model_dump()

    # Step 1: Use the classification logic (returns dict with reason_summary)
    triage_result = test_script.classify_patient(vitals_dict)
    classification = triage_result["classification"]
    reason_summary = triage_result["reason_summary"] # Capture the pre-made reason

    # Step 2: Reasoning-style AI prompt
    # We ask the model to rewrite the input line using a standard instruction format.
    ai_prompt = f"""
    Combine the following two pieces of information into one concise medical summary sentence:
    Reason: "{reason_summary}"
    Classification: "{classification}"
    
    Summary:
    """

    # Step 3: Use the local model to generate the explanation asynchronously
    # We are setting a low max_tokens and stopping on newline.
    output = await asyncio.to_thread(llm, ai_prompt, max_tokens=120, stop=['\n'])
    explanation = output["choices"][0]["text"].strip()

    # Step 4: Return the result
    return {
        "classification": classification,
        "ai_reasoning": explanation
    }

# ------------------- API ENDPOINT: Serve Frontend -------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the simple HTML interface when a user accesses the root URL.
    """
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend HTML file not found!</h1>", status_code=404)

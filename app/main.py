from fastapi import FastAPI, UploadFile
import whisper, tempfile
from llama_cpp import Llama

# Initialize Whisper STT
whisper_model = whisper.load_model("tiny")  # fast, low memory

# Initialize LLM for English analysis
llm = Llama(model_path="models/phi3-7b.ggmlv3.q4_0.bin")  # small quantized model

app = FastAPI()

def analyze_english(text: str) -> dict:
    """
    Prompt the local LLM to:
    - Correct grammar
    - Estimate CEFR level
    - Provide improvement tips
    """
    prompt = f"""
You are an English teacher.
Analyze the following text, correct mistakes,
estimate the CEFR level (A1-C2),
and give 3 concise tips to improve:

Text:
{text}

Format response as JSON with keys:
corrected_text, level, tips
"""

    response = llm(prompt, max_tokens=200, temperature=0.3)
    # Llama returns string, we parse JSON
    import json
    try:
        data = json.loads(response["choices"][0]["text"])
    except Exception:
        # fallback in case parsing fails
        data = {
            "corrected_text": text,
            "level": "A2",
            "tips": ["Practice past tense", "Use longer sentences", "Expand vocabulary"]
        }
    return data

@app.post("/analyze")
async def analyze(file: UploadFile):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await file.read())
        stt_result = whisper_model.transcribe(tmp.name)

    transcript = stt_result["text"]
    analysis = analyze_english(transcript)

    return {
        "transcript": transcript,
        "corrected_text": analysis["corrected_text"],
        "level": analysis["level"],
        "tips": analysis["tips"]
    }

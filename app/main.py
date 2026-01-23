from fastapi import FastAPI, UploadFile
import whisper, tempfile
from llama_cpp import Llama
import json

app = FastAPI()

# Initialize Whisper Tiny
whisper_model = whisper.load_model("tiny")  # fast and light

# Initialize LLM
llm = Llama(model_path="models/llama-7b.ggmlv3.q4_0.bin")  # your model

def analyze_english(text: str) -> dict:
    prompt = f"""
You are an English teacher.
Correct the grammar, estimate CEFR level (A1-C2),
and give 3 concise improvement tips for the following text:

{text}

Return a JSON object with keys:
corrected_text, level, tips
"""
    response = llm(prompt, max_tokens=200, temperature=0.3)
    try:
        data = json.loads(response["choices"][0]["text"])
    except Exception:
        data = {
            "corrected_text": text,
            "level": "A2",
            "tips": ["Practice past tense", "Use longer sentences", "Expand vocabulary"]
        }
    return data

@app.post("/analyze")
async def analyze(file: UploadFile):
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

from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
import whisper
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

# ------------------ MODEL LOADER ------------------
model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("tiny", device="cpu")
    return model

# ------------------ CONFIG ------------------
API_SECRET = os.getenv("API_SECRET")

# ------------------ SCHEMA ------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ------------------ ENDPOINT ------------------
@app.post("/api/voice-detection")
async def detect_voice(req: Request, body: VoiceRequest):

    api_key = req.headers.get("x-api-key") or req.headers.get("x_api_key")
    print("VERSION 4 DEPLOYED")
    print("HEADERS:", dict(req.headers))

    # -------- API KEY --------
    if not API_SECRET:
        return {"status": "error", "message": "Server misconfiguration: API secret missing"}

    if api_key != API_SECRET:
        return {"status": "error", "message": "Invalid API key"}

    # -------- INPUT VALIDATION --------
    if body.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": "Unsupported language"}

    if body.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format"}

    # -------- AUDIO PROCESSING --------
    try:
        audio_bytes = base64.b64decode(body.audioBase64)
    except Exception:
        return {"status": "error", "message": "Invalid base64 audio"}

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name
    except Exception:
        return {"status": "error", "message": "Failed saving audio"}

    try:
        result = get_model().transcribe(temp_path)
        transcript = result["text"]
    except Exception:
        os.remove(temp_path)
        return {"status": "error", "message": "Speech transcription failed"}

    try:
        y, sr = librosa.load(temp_path, sr=16000)
    except Exception:
        os.remove(temp_path)
        return {"status": "error", "message": "Audio decoding failed"}

    os.remove(temp_path)

    # -------- FEATURE EXTRACTION --------
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_variance = float(np.var(pitch))
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # -------- DETECTION LOGIC --------
    audio_score = 0
    text_score = 0

    if pitch_variance < 50:
        audio_score += 0.3
    else:
        text_score += 0.1

    if zcr_mean < 0.05:
        audio_score += 0.3
    else:
        text_score += 0.1

    if spectral_flatness < 0.01:
        audio_score += 0.3
    else:
        text_score += 0.1

    lower_transcript = transcript.lower()
    if len(lower_transcript.split()) > 15 and all(w not in lower_transcript for w in ["uh", "um", "hmm"]):
        audio_score += 0.2
    else:
        text_score += 0.1

    confidence = min(audio_score + text_score, 1.0)
    score_gap = abs(audio_score - text_score)

    if score_gap < 0.15:
        confidence *= 0.7
        ambiguity_flag = True
    elif score_gap < 0.30:
        confidence *= 0.85
        ambiguity_flag = True
    else:
        ambiguity_flag = False

    if audio_score > text_score:
        classification = "AI_GENERATED"
        explanation = "Low pitch variance and smooth spectral features suggest synthetic voice"
    else:
        classification = "HUMAN"
        explanation = "Natural waveform irregularities indicate human speech"

    if ambiguity_flag:
        explanation += ". Mixed signal traits reduce confidence."

    return {
        "status": "success",
        "language": body.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }

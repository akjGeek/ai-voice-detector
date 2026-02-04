from fastapi import FastAPI, Header
from pydantic import BaseModel
import base64
import whisper
import librosa
import numpy as np
import tempfile
import os
import time

app = FastAPI()

# ---------------- MODEL LOADING ----------------
model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("tiny", device="cpu")
    return model

@app.on_event("startup")
def preload_model():
    get_model()
    print("Model preloaded successfully")

# ---------------- SECURITY ----------------
API_SECRET = os.getenv("API_SECRET")

# ---------------- REQUEST MODEL ----------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
LANGUAGE_MAP = {
    "Tamil": "ta",
    "English": "en",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Telugu": "te"
}

# ---------------- API ENDPOINT ----------------
@app.post("/api/voice-detection")
async def detect_voice(
    body: VoiceRequest,
    x_api_key: str = Header(None)
):
    print("FINAL VERSION DEPLOYED")

    # ---- Server configuration check ----
    if not API_SECRET:
        return {"status": "error", "message": "Server misconfiguration: API secret missing"}

    # ---- API Key validation ----
    if x_api_key != API_SECRET:
        return {"status": "error", "message": "Invalid API key"}

    # ---- Language validation ----
    if body.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": "Unsupported language"}

    # ---- Format validation ----
    if body.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format"}

    # ---- Short input check ----
    if len(body.audioBase64) < 100:
        return {"status": "error", "message": "Audio too short or invalid"}

    # ---- Decode Base64 safely ----
    try:
        audio_bytes = base64.b64decode(body.audioBase64 + "==")
    except Exception:
        return {"status": "error", "message": "Invalid base64 audio"}

    # ---- Size limit (protect RAM) ----
    if len(audio_bytes) > 5 * 1024 * 1024:
        return {"status": "error", "message": "Audio file too large"}

    temp_path = None
    start_time = time.time()

    try:
        # ---- Save temp file ----
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        # ---- Transcription ----
        whisper_model = get_model()
        result = whisper_model.transcribe(temp_path, language=LANGUAGE_MAP[body.language])
        transcript = result["text"].strip()

        # ---- Audio Feature Extraction ----
        y, sr = librosa.load(temp_path, sr=16000, mono=False)
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        if len(y) == 0:
            return {"status": "error", "message": "Empty audio file"}

        yin_pitches = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        pitch_var = float(np.var(yin_pitches))

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        stft = np.abs(librosa.stft(y))
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=stft)))

        # ---- Timeout protection ----
        if time.time() - start_time > 20:
            return {"status": "error", "message": "Processing timeout"}

        # ---- Scoring ----
        audio_score = 0.0
        text_score = 0.0

        if pitch_var < 100:
            audio_score += 0.3
        else:
            text_score += 0.1

        if zcr < 0.08:
            audio_score += 0.3
        else:
            text_score += 0.1

        if flatness > 0.15:
            audio_score += 0.3
        else:
            text_score += 0.1

        word_count = len(transcript.split())
        fillers = ["uh", "um", "hmm", "er", "ah"]
        if word_count > 15 and not any(w in transcript.lower() for w in fillers):
            text_score += 0.2
        else:
            audio_score += 0.1

        confidence = min(audio_score + text_score, 1.0)
        score_gap = abs(audio_score - text_score)

        if score_gap < 0.15:
            confidence *= 0.7
        elif score_gap < 0.3:
            confidence *= 0.85

        if audio_score > text_score + 0.05:
            classification = "AI_GENERATED"
            explanation = "Synthetic speech characteristics detected"
        else:
            classification = "HUMAN"
            explanation = "Natural speech characteristics detected"

        return {
            "status": "success",
            "language": body.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }


    except Exception as e:
        print("ERROR:", str(e))
        return {
            "status": "error",
            "message": "Processing failed",
            "details": str(e)[:100]
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

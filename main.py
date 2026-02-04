from fastapi import FastAPI, Header
from pydantic import BaseModel
import base64
import whisper
import librosa
import numpy as np
import tempfile
import os
import binascii

app = FastAPI()
print("VERSION 13 DEPLOYED")
# Load Whisper lazily to save memory
model = None
def get_model():
    global model
    if model is None:
        model = whisper.load_model("tiny", device="cpu")
    return model

API_SECRET = os.getenv("API_SECRET")

# Request schema
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

@app.post("/api/voice-detection")
async def detect_voice(body: VoiceRequest, x_api_key: str = Header(None)):

    # 🔐 Server config validation
    if not API_SECRET:
        return {"status": "error", "message": "Server misconfiguration: API secret missing"}

    # 🔑 API key validation
    if x_api_key != API_SECRET:
        return {"status": "error", "message": "Invalid API key"}

    # 🌍 Language validation
    if body.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": "Unsupported language"}

    # 🎵 Audio format validation
    if body.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format"}

    try:
        # 📦 Safe Base64 decoding
        try:
            audio_bytes = base64.b64decode(body.audioBase64, validate=True)
        except binascii.Error:
            return {"status": "error", "message": "Invalid Base64 audio"}

        # 💾 Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        # 📝 Transcription (lightweight)
        whisper_model = get_model()
        result = whisper_model.transcribe(temp_path, language=LANGUAGE_MAP[body.language])
        transcript = result["text"].strip()

        # 🎧 Load audio safely
        y, sr = librosa.load(temp_path, sr=16000, mono=False)
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        if len(y) == 0:
            raise ValueError("Empty audio")

        # 🎼 Feature extraction
        yin_vals = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        pitch_var = float(np.var(yin_vals))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        stft = np.abs(librosa.stft(y))
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=stft)))

        os.remove(temp_path)

        # 🧠 AI vs Human Scoring
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

        # Minimal transcript influence
        if len(transcript.split()) > 10:
            text_score += 0.1

        confidence = min(audio_score + text_score, 1.0)
        score_gap = abs(audio_score - text_score)

        if score_gap < 0.15:
            confidence *= 0.7
        elif score_gap < 0.3:
            confidence *= 0.85

        # 🎯 Final Decision
        if audio_score > text_score:
            classification = "AI_GENERATED"
            explanation = f"Low pitch variance ({pitch_var:.1f}) and flat spectral profile suggest synthetic speech"
        else:
            classification = "HUMAN"
            explanation = f"High pitch variation and natural spectral dynamics indicate human speech"

        # ✅ EXACT FORMAT REQUIRED BY CHALLENGE
        return {
            "status": "success",
            "language": body.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

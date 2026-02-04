from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
import base64
import whisper
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

# Lazy Whisper load (prevents memory spike at boot)
model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("tiny", device="cpu")
    return model

API_SECRET = os.getenv("API_SECRET")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
LANGUAGE_MAP = {
    "Tamil": "ta", "English": "en", "Hindi": "hi",
    "Malayalam": "ml", "Telugu": "te"
}

@app.post("/api/voice-detection")
async def detect_voice(
    body: VoiceRequest,
    x_api_key: str = Header(None)
):
    print("VERSION 12 DEPLOYED")

    # 🔐 Server config check
    if not API_SECRET:
        return {"status": "error", "message": "Server misconfiguration: API secret missing"}

    # 🔑 API key validation
    if x_api_key != API_SECRET:
        return {"status": "error", "message": "Invalid API key"}

    # 🌍 Language check
    if body.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": "Unsupported language"}

    # 🎵 Format check
    if body.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format"}

    try:
        # Decode Base64 audio
        audio_bytes = base64.b64decode(body.audioBase64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        # Transcription with language hint
        whisper_model = get_model()
        result = whisper_model.transcribe(temp_path, language=LANGUAGE_MAP[body.language])
        transcript = result["text"].strip()

        # Audio loading
        y, sr = librosa.load(temp_path, sr=16000, mono=False)

        # Stereo → mono
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        if len(y) == 0:
            raise ValueError("Empty audio file")

        # ===== FEATURE EXTRACTION =====

        # Pitch variance
        yin_pitches = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        pitch_var = float(np.var(yin_pitches))

        # Zero Crossing Rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Spectral flatness
        stft = np.abs(librosa.stft(y))
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=stft)))

        # 🔥 NEW FEATURE — Energy variance (major improvement)
        energy = librosa.feature.rms(y=y)[0]
        energy_var = float(np.var(energy))

        os.remove(temp_path)

        # ===== SCORING SYSTEM =====

        audio_score = 0.0
        text_score = 0.0

        # Pitch variance
        if pitch_var < 100:
            audio_score += 0.3
        else:
            text_score += 0.1

        # ZCR
        if zcr < 0.08:
            audio_score += 0.3
        else:
            text_score += 0.1

        # Spectral flatness
        if flatness > 0.15:
            audio_score += 0.3
        else:
            text_score += 0.1

        # 🔥 Energy variance scoring
        if energy_var < 0.001:
            audio_score += 0.2
        else:
            text_score += 0.1

        # Transcript behavior
        word_count = len(transcript.split())
        fillers = ["uh", "um", "hmm", "er", "ah"]
        has_few_fillers = not any(w in transcript.lower() for w in fillers)

        if word_count > 15 and has_few_fillers:
            text_score += 0.2
        else:
            audio_score += 0.1

        confidence = min(audio_score + text_score, 1.0)
        score_gap = abs(audio_score - text_score)

        # Reduce confidence if mixed signals
        if score_gap < 0.15:
            confidence *= 0.7
        elif score_gap < 0.3:
            confidence *= 0.85

        # Final classification
        if audio_score > text_score + 0.05:
            classification = "AI_GENERATED"
            explanation = "Synthetic speech characteristics detected (low variance, flat spectrum, stable energy)"
        else:
            classification = "HUMAN"
            explanation = "Natural speech characteristics detected (high variance, natural fluctuations)"

        return {
            "status": "success",
            "language": body.language,
            "transcriptPreview": transcript[:200] + "..." if len(transcript) > 200 else transcript,
            "wordCount": word_count,
            "features": {
                "pitchVar": round(pitch_var, 2),
                "zcr": round(zcr, 3),
                "flatness": round(flatness, 3),
                "energyVar": round(energy_var, 5)
            },
            "audioScore": round(audio_score, 2),
            "textScore": round(text_score, 2),
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "details": type(e).__name__
        }

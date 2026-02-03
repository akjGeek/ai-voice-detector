from fastapi import FastAPI, Header, HTTPException
from fastapi import Request
from pydantic import BaseModel
import base64
import whisper
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

# Load Whisper once 
model = None
print("VERSION 2 DEPLOYED")

def get_model():
    global model
    if model is None:
        model = whisper.load_model("tiny", device="cpu")
    return model

#model = whisper.load_model("base")
#API_SECRET = "my_secret_key"

API_SECRET = os.getenv("API_SECRET")

# Request Schema
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

@app.post("/api/voice-detection")
async def detect_voice(request: Request, body: VoiceRequest):

    api_key = request.headers.get("x-api-key")

    if api_key != API_SECRET:
        return {"status": "error", "message": "Invalid API key"}

    if request.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": "Unsupported language"}

    if request.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format"}

    try:
        # Decode Base64
        audio_bytes = base64.b64decode(request.audioBase64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        # Transcription
        #result = model.transcribe(temp_path)
        result = get_model().transcribe(temp_path)

        transcript = result["text"]

        # Audio Features
        #y, sr = librosa.load(temp_path)
        y, sr = librosa.load(temp_path, sr=16000)

        mfcc_mean = float(np.mean(librosa.feature.mfcc(y=y, sr=sr)))
        pitch_mean = float(np.mean(librosa.yin(y, fmin=50, fmax=300)))

        os.remove(temp_path)
        # Additional forensic features
        pitch_variance = float(np.var(librosa.yin(y, fmin=50, fmax=300)))
        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        # -------- Advanced Detection Logic --------
        audio_score = 0
        text_score = 0


        # Pitch variance (AI voices often too consistent)
        if pitch_variance < 50:
            audio_score += 0.3
        else:
            text_score += 0.1

        # Zero-crossing rate (synthetic audio cleaner)
        if zcr_mean < 0.05:
            audio_score += 0.3
        else:
            text_score += 0.1

        # Spectral flatness (AI lacks natural noise)
        if spectral_flatness < 0.01:
            audio_score += 0.3
        else:
            text_score += 0.1

        # Transcript perfection
        lower_transcript = transcript.lower()
        if len(lower_transcript.split()) > 15 and all(w not in lower_transcript for w in ["uh","um","hmm"]):
            audio_score += 0.2
        else:
            text_score += 0.1

        confidence = min(audio_score + text_score, 1.0)

        # -------- Ambiguity Detection --------
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
            explanation = "Low pitch variance, smooth waveform structure, and reduced natural noise suggest synthetic voice"
        else:
            classification = "HUMAN"
            explanation = "Irregular waveform dynamics and natural noise patterns indicate human speech"

        # Add ambiguity note
        if ambiguity_flag:
            explanation += ". However, signal characteristics show overlap between human and AI traits, reducing confidence."

            return {
                "status": "success",
                "language": request.language,
                "classification": classification,
                "confidenceScore": confidence,
                "explanation": explanation
                }

    except Exception as e:
        return {"status": "error", "message": str(e)}


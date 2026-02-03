from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64, whisper, librosa, numpy as np, tempfile, os

app = FastAPI()

model = None
print("VERSION 8 DEPLOYED")
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

@app.post("/api/voice-detection")
async def detect_voice(request: Request, body: VoiceRequest):

    print("VERSION 8 DEPLOYED")

    # ✅ DEFINE API KEY PROPERLY
    api_key = request.headers.get("x-api-key") or request.headers.get("x_api_key")

    if not API_SECRET:
        return {"status": "error", "message": "Server misconfiguration: API secret missing"}

    if api_key != API_SECRET:
        return {"status": "error", "message": "Invalid API key"}

    # ✅ USE body, NOT request
    if body.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": "Unsupported language"}

    if body.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format"}

    try:
        audio_bytes = base64.b64decode(body.audioBase64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        result = get_model().transcribe(temp_path)
        transcript = result["text"]

        y, sr = librosa.load(temp_path, sr=16000)

        pitch_var = float(np.var(librosa.yin(y, fmin=50, fmax=300)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

        os.remove(temp_path)

        audio_score = 0
        text_score = 0

        if pitch_var < 50: audio_score += 0.3
        else: text_score += 0.1

        if zcr < 0.05: audio_score += 0.3
        else: text_score += 0.1

        if flatness < 0.01: audio_score += 0.3
        else: text_score += 0.1

        if len(transcript.split()) > 15 and not any(w in transcript.lower() for w in ["uh","um","hmm"]):
            audio_score += 0.2
        else:
            text_score += 0.1

        confidence = min(audio_score + text_score, 1.0)
        score_gap = abs(audio_score - text_score)

        if score_gap < 0.15: confidence *= 0.7
        elif score_gap < 0.3: confidence *= 0.85

        if audio_score > text_score:
            classification = "AI_GENERATED"
            explanation = "Synthetic speech characteristics detected"
        else:
            classification = "HUMAN"
            explanation = "Natural speech characteristics detected"

        return {
            "status": "success",
            "language": body.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}



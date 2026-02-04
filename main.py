from fastapi import FastAPI, Header
from pydantic import BaseModel
import base64, whisper, librosa, numpy as np, tempfile, os

app = FastAPI()

# =========================
# 🔧 ENVIRONMENT CONFIG
# =========================
API_SECRET = os.getenv("API_SECRET")
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

print("API STARTED")
print("TEST MODE:", TEST_MODE)

# =========================
# 🎤 LOAD WHISPER MODEL ONCE
# =========================
model = None
def get_model():
    global model
    if model is None:
        model = whisper.load_model("tiny", device="cpu")
    return model

# =========================
# 📦 REQUEST SCHEMA
# =========================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
LANGUAGE_MAP = {"Tamil":"ta","English":"en","Hindi":"hi","Malayalam":"ml","Telugu":"te"}

# =========================
# 🚀 API ENDPOINT
# =========================
@app.post("/api/voice-detection")
async def detect_voice(body: VoiceRequest, x_api_key: str = Header(None)):

    if not API_SECRET:
        return {"status":"error","message":"Server misconfiguration: API secret missing"}

    if x_api_key != API_SECRET:
        return {"status":"error","message":"Invalid API key"}

    if body.language not in SUPPORTED_LANGUAGES:
        return {"status":"error","message":"Unsupported language"}

    if body.audioFormat.lower() != "mp3":
        return {"status":"error","message":"Invalid audio format"}

    try:
        # =========================
        # 🎵 Decode audio
        # =========================
        audio_bytes = base64.b64decode(body.audioBase64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        # =========================
        # 🗣 Speech-to-text
        # =========================
        whisper_model = get_model()
        result = whisper_model.transcribe(temp_path, language=LANGUAGE_MAP[body.language])
        transcript = result["text"].strip()

        # =========================
        # 🎼 Audio Loading
        # =========================
        if TEST_MODE:
            y, sr = librosa.load(temp_path, sr=16000, mono=True, duration=6)
        else:
            y, sr = librosa.load(temp_path, sr=16000, mono=True)

        if len(y) == 0:
            raise ValueError("Empty audio file")

        # =========================
        # 🔍 Feature Extraction
        # =========================

        # Pitch variance
        if TEST_MODE:
            yin_pitches = librosa.yin(y[:sr*4], fmin=70, fmax=250, sr=sr)
        else:
            yin_pitches = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        pitch_var = float(np.var(yin_pitches))

        # Zero Crossing Rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Spectral Flatness
        if TEST_MODE:
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        else:
            stft = np.abs(librosa.stft(y))
            flatness = float(np.mean(librosa.feature.spectral_flatness(S=stft)))

        os.remove(temp_path)

        # =========================
        # 🧠 SCORING LOGIC
        # =========================
        audio_score = 0.0
        text_score = 0.0

        if pitch_var < 100: audio_score += 0.3
        else: text_score += 0.1

        if zcr < 0.08: audio_score += 0.3
        else: text_score += 0.1

        if flatness > 0.15: audio_score += 0.3
        else: text_score += 0.1

        word_count = len(transcript.split())
        fillers = ["uh","um","hmm","er","ah"]
        if word_count > 15 and not any(w in transcript.lower() for w in fillers):
            text_score += 0.2
        else:
            audio_score += 0.1

        confidence = min(audio_score + text_score, 1.0)
        gap = abs(audio_score - text_score)

        if gap < 0.15: confidence *= 0.7
        elif gap < 0.3: confidence *= 0.85

        if audio_score > text_score + 0.05:
            classification = "AI_GENERATED"
            explanation = "Synthetic speech characteristics detected"
        else:
            classification = "HUMAN"
            explanation = "Natural speech characteristics detected"

        # =========================
        # ✅ RESPONSE
        # =========================
        return {
            "status": "success",
            "language": body.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }
    except Exception as e:
        return {"status":"error","message":str(e),"details":type(e).__name__}

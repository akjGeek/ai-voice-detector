import base64

with open("sample.mp3", "rb") as audio_file:
    encoded = base64.b64encode(audio_file.read()).decode("utf-8")

with open("base64.txt", "w") as f:
    f.write(encoded)

print("Saved to base64.txt")

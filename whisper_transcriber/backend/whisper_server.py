from flask import Flask, request, jsonify
import tempfile
from faster_whisper import WhisperModel

app = Flask(__name__)
model = WhisperModel("base", device="cpu", compute_type="int8")  # Or "small", etc.

@app.route("/transcribe", methods=["POST"])
def transcribe():
    print("Received audio:", request.files.get("audio"))
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio.save(tmp.name)
        segments, _ = model.transcribe(tmp.name)
        text = " ".join([seg.text for seg in segments])
    print("Transcription:", text)
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

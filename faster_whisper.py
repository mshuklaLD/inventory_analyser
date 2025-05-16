# whisper_server.py using faster-whisper
from flask import Flask, request, jsonify
import tempfile
from faster_whisper import WhisperModel

app = Flask(__name__)
model = WhisperModel("base")  # Or "small", "medium", "large-v3"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio.save(tmp.name)
        segments, _ = model.transcribe(tmp.name)

        # Combine all segments into one transcript
        transcription = " ".join([segment.text for segment in segments])
        return jsonify({"text": transcription})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

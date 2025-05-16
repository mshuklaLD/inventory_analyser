FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg && \
    pip install --no-cache-dir flask faster-whisper

COPY whisper_server.py .

CMD ["python", "whisper_server.py"]

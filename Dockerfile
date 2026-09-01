FROM python:3.11-slim

# System ffmpeg + fonts (matches the fonts bundled for the desktop app, so
# rendering looks identical) — real apt access here, so no need for a
# bundled ffmpeg binary like the sandboxed macOS launch path uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fontconfig fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY frontend frontend

RUN mkdir -p data

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

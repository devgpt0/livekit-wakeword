FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 sox espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY configs/ configs/

RUN uv sync --frozen --extra train --extra export --no-dev \
    || uv sync --extra train --extra export --no-dev

ENTRYPOINT ["uv", "run", "livewakeword"]

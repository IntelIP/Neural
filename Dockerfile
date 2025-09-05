FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Optional: install uv for reproducible installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (leverage caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev || pip install .

# Copy source and configs
COPY neural_sdk ./neural_sdk
COPY config ./config

# Default command does a smoke import
CMD ["python", "-c", "import neural_sdk; print('Neural SDK OK')"]


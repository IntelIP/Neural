# Neural SDK Team Docker Image
# Version: 1.1.0 - WebSocket Streaming

FROM python:3.11-slim

# Install git for private repo access
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Neural SDK v1.1.0
RUN pip install git+https://github.com/IntelIP/Neural-Trading-Platform.git@v1.1.0

# Copy your trading application
COPY . .

# Install additional requirements if needed
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Default command
CMD ["python", "main.py"]
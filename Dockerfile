FROM huggingface/transformers-pytorch-gpu:latest

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers \
    torchaudio \
    soundfile \
    python-multipart

# Copy application code
COPY server.py /app/server.py

# Expose the port (purely informational, still configurable)
EXPOSE 8000

# Start the server
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8000"]

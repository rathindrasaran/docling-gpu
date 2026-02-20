# Use a RunPod base image with PyTorch and CUDA support
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# 1. Install required system dependencies, adding build tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip to handle complex dependency resolution efficiently
RUN pip install --no-cache-dir --upgrade pip

# 3. Split installations to prevent Out-Of-Memory (OOM) kills during build
RUN pip install --no-cache-dir runpod docling transformers accelerate

# Pre-fetch the SmolDocling model to minimize cold start latency
RUN docling-tools models download-hf-repo ds4sd/SmolDocling-256M-preview || true

# Copy your serverless handler script into the container
COPY handler.py /handler.py

# Set the entrypoint to start the RunPod serverless worker
CMD [ "python", "-u", "/handler.py" ]

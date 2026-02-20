# Use a RunPod base image with PyTorch and CUDA support
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install required system dependencies for OpenCV and Docling
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK and Docling ecosystem
RUN pip install --no-cache-dir runpod docling docling-tools transformers accelerate

# Pre-fetch the SmolDocling model to minimize cold start latency
RUN docling-tools models download-hf-repo ds4sd/SmolDocling-256M-preview || true

# Copy your serverless handler script into the container
COPY handler.py /handler.py

ENTRYPOINT []
# Set the entrypoint to start the RunPod serverless worker
CMD [ "python", "-u", "/handler.py" ]

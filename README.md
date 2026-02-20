# Serverless PDF to Markdown via SmolDocling on RunPod

This repository contains the necessary configuration to deploy a high-throughput, serverless PDF-to-Markdown conversion endpoint on RunPod. It utilizes IBM's **SmolDocling-256M-preview** Vision-Language Model (VLM) for end-to-end document parsing.



## Features
* **VLM-Based Parsing:** Bypasses traditional OCR heuristics, providing superior layout understanding and table extraction.
* **Optimized Cold Starts:** The `Dockerfile` pre-fetches the SmolDocling weights from Hugging Face during the build process.
* **High-Concurrency Async Handler:** Specifically tuned for high-VRAM nodes (e.g., RTX 4090 24GB). It uses `asyncio` to process multiple documents concurrently without blocking the RunPod event loop, maximizing GPU utilization.

## Prerequisites
* Docker installed locally or in your CI/CD pipeline.
* A [RunPod](https://www.runpod.io/) account with API keys configured.
* A container registry (Docker Hub, GitHub Container Registry, etc.) to host your built image.

## Project Structure
* `Dockerfile`: Handles system dependencies, the RunPod SDK, Docling, and model pre-fetching.
* `handler.py`: The asynchronous serverless entry point that manages the document conversion pipeline and concurrency limits.

## Deployment Steps

### 1. Build and Push the Docker Image
Build the image locally and push it to your preferred container registry:

```bash
# Build the image
docker build -t yourusername/smoldocling-runpod:latest .

# Push to your registry
docker push yourusername/smoldocling-runpod:latest
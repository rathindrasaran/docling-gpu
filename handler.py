import os
import base64
import tempfile
import asyncio
import runpod
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, AcceleratorOptions
from docling.pipeline.vlm_pipeline import VlmPipeline

pipeline_options = VlmPipelineOptions()

# Lower the thread count per inference slightly. 
# With 10 concurrent requests, we don't want to oversubscribe the CPU cores.
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=2, 
    device="cuda" 
)

from docling.datamodel import vlm_model_specs

# Explicitly assign the Transformers backend for NVIDIA CUDA execution
pipeline_options.vlm_options = vlm_model_specs.SMOLDOCLING_TRANSFORMERS

# Initialize globally to cache in VRAM
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

def process_document(file_path):
    """Synchronous CPU/GPU bound conversion task."""
    result = doc_converter.convert(file_path)
    return result.document.export_to_markdown()

async def handler(job):
    """Asynchronous handler to accept concurrent RunPod jobs."""
    job_input = job['input']
    pdf_base64 = job_input.get('pdf_base64')

    if not pdf_base64:
        return {"error": "Invalid payload: Missing 'pdf_base64'."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(base64.b64decode(pdf_base64))
        tmp_file_path = tmp_file.name

    try:
        # Offload the blocking inference to a separate thread pool.
        # This keeps the main RunPod event loop free to accept the next concurrent job.
        markdown_output = await asyncio.to_thread(process_document, tmp_file_path)
        return {"markdown": markdown_output}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

if __name__ == '__main__':
    # An RTX 4090 (24GB) can easily hold 10+ instances of a 256M model.
    # We statically set the concurrency modifier to 10.
    runpod.serverless.start({
        'handler': handler,
        'concurrency_modifier': lambda current_concurrency: 10 
    })

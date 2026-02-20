import os
import base64
import tempfile
import runpod
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, AcceleratorOptions
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel import vlm_model_specs

# 1. Configure the pipeline for single-thread execution per worker
pipeline_options = VlmPipelineOptions()

# Enforce CUDA and allow it to use more CPU threads now that it isn't sharing resources
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4, 
    device="cuda" 
)

# Use the Transformers backend for NVIDIA execution
pipeline_options.vlm_options = vlm_model_specs.SMOLDOCLING_TRANSFORMERS

# 2. Initialize the converter globally
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

def handler(job):
    """Synchronous RunPod Serverless Handler."""
    job_input = job.get('input', {})
    pdf_base64 = job_input.get('pdf_base64')

    if not pdf_base64:
        return {"error": "Invalid payload: Missing 'pdf_base64'."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(base64.b64decode(pdf_base64))
        tmp_file_path = tmp_file.name

    try:
        # Run inference synchronously
        result = doc_converter.convert(tmp_file_path)
        return {"markdown": result.document.export_to_markdown()}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})

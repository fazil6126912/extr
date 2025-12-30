import os
import json
import base64
import argparse
import tempfile
import time  # For latency
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path

import streamlit as st  # For web UI; install with `pip install streamlit`
from dotenv import load_dotenv
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
import fitz  # PyMuPDF for rendering pages to images
from langsmith import Client  # For LangSmith tracing

# Load environment variables
load_dotenv(override=True)

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Pricing dict (Standard tier, per 1M tokens; expanded with requested + extras)
MODEL_PRICING = {
    # Requested
    'gpt-5.2': {'input': 1.75, 'output': 14.00},
    'gpt-5': {'input': 1.25, 'output': 10.00},
    'gpt-5-mini': {'input': 0.25, 'output': 2.00},
    'gpt-4.1': {'input': 2.00, 'output': 8.00},
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
    # Extras (for more testing)
    'gpt-5-nano': {'input': 0.05, 'output': 0.40},
    'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    'o1': {'input': 15.00, 'output': 60.00},
    'o1-mini': {'input': 1.10, 'output': 4.40},
}

VISION_MODELS = [
    'gpt-5.2', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
    'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
    'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'
]  # All gpt-4+ and gpt-5 series support vision (as of 2025)

# Prompts (unchanged)
EXTRACTION_PROMPT_TEXT = """Analyze the following document content thoroughly. Extract all relevant details, entities, fields, and information as key-value pairs in a JSON object. 

Choose appropriate keys based on the content (e.g., for a form: "name": "John Doe", "date": "2023-01-01"; for an invoice: "invoice_number": "INV-123", "total_amount": 100.50). 
Include nested objects if needed (e.g., {{"address": {{"street": "...", "city": "..."}}}}). 
Be comprehensive but concise—only include meaningful data. Output ONLY valid JSON, no additional text."""

EXTRACTION_PROMPT_VISION = """Analyze the following document or image content thoroughly (this may be multi-page). Extract all relevant details, entities, fields, and information as key-value pairs in a JSON object. 

Choose appropriate keys based on the content (e.g., for a form: "name": "John Doe", "date": "2023-01-01"; for an invoice: "invoice_number": "INV-123", "total_amount": 100.50; for a report: "patient_id": "...", "results": {...}). 
Include nested objects/arrays if needed (e.g., {{"address": {{"street": "...", "city": "..."}}}} or {{"test_results": [{"test": "Karyotype", "value": "..."}]}}). 
Be comprehensive but concise—only include meaningful data. Output ONLY valid JSON, no additional text."""

parser = JsonOutputParser()

def clean_markdown(content: str) -> str:
    """Strip markdown fencing like ```json ... ``` for clean JSON parsing."""
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned

def get_llm(model_name: str):
    """Initialize LLM with selected model."""
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Unsupported model: {model_name}")
    return ChatOpenAI(model=model_name, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

def calculate_cost(usage_metadata, model_name: str) -> tuple[float, float]:
    """Calculate cost in USD and INR based on usage_metadata."""
    if not usage_metadata:
        return 0.0, 0.0
    input_tokens = usage_metadata.get('input_tokens', 0)
    output_tokens = usage_metadata.get('output_tokens', 0)
    input_price = MODEL_PRICING[model_name]['input'] / 1_000_000
    output_price = MODEL_PRICING[model_name]['output'] / 1_000_000
    usd = (input_tokens * input_price) + (output_tokens * output_price)
    inr = usd * 90
    return usd, inr

def extract_from_pdf(file_path: str, llm, is_vision: bool) -> tuple[dict, dict, float]:
    """Extract from PDF: Text or vision mode, return result, usage_metadata, latency."""
    start_time = time.time()
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    
    if text.strip() and not is_vision:
        # Text mode: Invoke LLM first to capture usage, then parse
        prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT_TEXT + "\n\nContent:\n{text}")
        messages = prompt.format_messages(text=text)
        response = llm.invoke(messages)
        cleaned = clean_markdown(response.content)
        result = json.loads(cleaned)
        usage_metadata = response.usage_metadata
    else:
        # Vision mode (render pages)
        if not is_vision:
            raise ValueError("Scanned PDF requires vision model. Select a gpt-4+ or gpt-5 model.")
        doc = fitz.open(file_path)
        image_contents = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
            })
        doc.close()
        
        if not image_contents:
            raise ValueError("No pages found in PDF.")
        
        message = HumanMessage(content=[{"type": "text", "text": EXTRACTION_PROMPT_VISION}] + image_contents)
        response = llm.invoke([message])
        cleaned = clean_markdown(response.content)
        result = json.loads(cleaned)
        usage_metadata = response.usage_metadata
    
    latency = time.time() - start_time
    return result, usage_metadata, latency

def extract_from_image(file_path: str, llm) -> tuple[dict, dict, float]:
    """Extract from image: Vision only."""
    if llm.model_name not in VISION_MODELS:
        raise ValueError("Image requires vision model. Select a gpt-4+ or gpt-5 model.")
    start_time = time.time()
    with open(file_path, "rb") as image_file:
        img_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    mime_type, _ = guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("Not a supported image.")
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": EXTRACTION_PROMPT_VISION},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_data}"}},
        ],
    )
    response = llm.invoke([message])
    cleaned = clean_markdown(response.content)
    result = json.loads(cleaned)
    
    latency = time.time() - start_time
    usage_metadata = response.usage_metadata
    return result, usage_metadata, latency

def main():
    # CLI (with model arg)
    parser = argparse.ArgumentParser(description="Extract structured JSON from PDF or image using LangChain + OpenAI.")
    parser.add_argument("file_path", type=str, help="Path to the input PDF or image file.")
    parser.add_argument("--output", "-o", type=str, default="extracted.json", help="Output JSON file path (default: extracted.json).")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o", choices=list(MODEL_PRICING.keys()), help="Model name (default: gpt-4o).")
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    llm = get_llm(args.model)
    ext = file_path.suffix.lower()
    is_vision = args.model in VISION_MODELS
    if ext == ".pdf":
        result, usage_metadata, latency = extract_from_pdf(str(file_path), llm, is_vision)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        result, usage_metadata, latency = extract_from_image(str(file_path), llm)
    else:
        raise ValueError(f"Unsupported file type: {ext}.")
    
    usd, inr = calculate_cost(usage_metadata, args.model)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Extraction complete! Results saved to {args.output}")
    print(json.dumps(result, indent=2))
    input_t = usage_metadata.get('input_tokens', 0)
    output_t = usage_metadata.get('output_tokens', 0)
    total_t = input_t + output_t
    print(f"Input tokens: {input_t:,} | Output tokens: {output_t:,} | Total tokens: {total_t:,} tokens / ${usd:.6f} / ₹{inr:.2f} | Latency: {latency:.2f}s")

# Enhanced Streamlit UI
def run_streamlit():
    st.title("🧠 GPT Model Extraction Tester")
    st.markdown("Upload a document (PDF/image) and select a model to test accuracy, tokens, cost, and latency. (15 models available)")
    
    # Model selection (searchable)
    models = list(MODEL_PRICING.keys())
    selected_model = st.selectbox("Select Model", models, index=models.index('gpt-4o'))
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "webp"])
    
    if uploaded_file and selected_model:
        if st.button("Extract & Analyze", type="primary"):
            with st.spinner(f"Extracting with {selected_model}..."):
                # Temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name
                
                try:
                    llm = get_llm(selected_model)
                    ext = Path(uploaded_file.name).suffix.lower()
                    is_vision = selected_model in VISION_MODELS
                    
                    if ext == ".pdf":
                        loader = PyMuPDFLoader(temp_path)
                        docs = loader.load()
                        text = "\n\n".join([doc.page_content for doc in docs])
                        text_len = len(text)
                        needs_vision = text_len == 0
                        if needs_vision and not is_vision:
                            st.warning("PDF is scanned (no text)—requires vision model. Falling back to text (may fail).")
                        result, usage_metadata, latency = extract_from_pdf(temp_path, llm, is_vision or needs_vision)
                    else:
                        result, usage_metadata, latency = extract_from_image(temp_path, llm)
                    
                    # Calculate cost
                    usd, inr = calculate_cost(usage_metadata, selected_model)
                    
                    # Display JSON
                    st.subheader("Extracted JSON")
                    st.json(result)
                    
                    # Download
                    st.download_button("Download JSON", json.dumps(result, indent=2), file_name=f"extracted_{selected_model}.json")
                    
                    # Metrics
                    input_t = usage_metadata.get('input_tokens', 0)
                    output_t = usage_metadata.get('output_tokens', 0)
                    total_t = input_t + output_t
                    col1, col2, col3 = st.columns(3)
                    col4, col5, col6 = st.columns(3)
                    with col1:
                        st.metric("Input Tokens", f"{input_t:,}")
                    with col2:
                        st.metric("Output Tokens", f"{output_t:,}")
                    with col3:
                        st.metric("Total Tokens", f"{total_t:,}")
                    with col4:
                        st.metric("Latency (s)", f"{latency:.2f}")
                    with col5:
                        st.metric("Total Cost in $", f"${usd:.8f}")
                    with col6:
                        st.metric("Total Cost in ₹", f"₹{inr:.2f}")
                    
                    # LangSmith trace note
                    st.caption(f"Traced in LangSmith project '{os.getenv('LANGCHAIN_PROJECT')}' for accuracy review.")
                    
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                finally:
                    os.unlink(temp_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != "run":
        main()
    else:
        run_streamlit()
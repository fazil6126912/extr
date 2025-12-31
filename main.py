import os
import json
import base64
import argparse
import tempfile
import time  # For latency
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import streamlit as st  # For web UI; install with `pip install streamlit`
from dotenv import load_dotenv
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from pydantic import BaseModel, Field
import fitz  # PyMuPDF for rendering pages to images
from langsmith import Client  # For LangSmith tracing

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage

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

# File types
FILE_TYPES = [
    "Clinical Notes", "Pathology", "Radiology", "Lab Results", 
    "Genomic Report", "Other"
]

# Type-specific schemas (Pydantic for structured output)
class TreatmentHistory(BaseModel):
    id: str = Field(..., description="Treatment ID")
    type: str = Field(..., description="Type e.g., surgery")
    name: str = Field(..., description="Treatment name")
    startDate: str = Field(..., description="Start date")
    endDate: str = Field(..., description="End date")
    status: str = Field(..., description="Status")
    response: str = Field(..., description="Response")
    responseDescription: str = Field(..., description="Response desc")
    notes: str = Field(..., description="Notes")

class FamilyHistory(BaseModel):
    relation: str = Field(..., description="Relation")
    cancerType: str = Field(..., description="Cancer type")
    ageAtDiagnosis: int = Field(..., description="Age at diagnosis")
    deceased: bool = Field(..., description="Deceased")
    ageAtDeath: int = Field(..., description="Age at death")

class Diagnosis(BaseModel):
    primary: str = Field(..., description="Primary diagnosis")
    icdCode: str = Field(..., description="ICD code")
    tnmStaging: dict = Field(..., description="TNM staging")
    dateOfDiagnosis: str = Field(..., description="Diagnosis date")
    confirmedBy: str = Field(..., description="Confirmed by")
    method: str = Field(..., description="Method")
    notes: str = Field(..., description="Notes")

class ClinicalNotesSchema(BaseModel):
    treatmentHistory: List[TreatmentHistory] = Field(..., description="Treatment history")
    familyHistory: List[FamilyHistory] = Field(..., description="Family history")
    diagnosis: Diagnosis = Field(..., description="Diagnosis")

class PathologySpecimen(BaseModel):
    specimenType: str = Field(..., description="Specimen type")
    collectionDate: str = Field(..., description="Collection date")
    histologicalType: str = Field(..., description="Histological type")
    grade: str = Field(..., description="Grade")
    margins: str = Field(..., description="Margins")
    lymphNodes: str = Field(..., description="Lymph nodes")
    tumorSize: str = Field(..., description="Tumor size")
    findings: List[str] = Field(..., description="Findings")
    ihcMarkers: List[dict] = Field(..., description="IHC markers")

class PathologySchema(BaseModel):
    pathology: List[PathologySpecimen] = Field(..., description="Pathology specimens")

class Biomarker(BaseModel):
    name: str = Field(..., description="Biomarker name")
    category: str = Field(..., description="Category")
    result: str = Field(..., description="Result")
    status: str = Field(..., description="Status")
    actionable: bool = Field(..., description="Actionable")
    clinicalRelevance: str = Field(..., description="Clinical relevance")
    therapeuticImplications: List[str] = Field(..., description="Therapeutic implications")
    testDate: str = Field(..., description="Test date")
    methodology: str = Field(..., description="Methodology")

class GenomicsSchema(BaseModel):
    biomarkers: List[Biomarker] = Field(..., description="Biomarkers")

class Imaging(BaseModel):
    id: str = Field(..., description="ID")
    modality: str = Field(..., description="Modality")
    bodyRegion: str = Field(..., description="Body region")
    date: str = Field(..., description="Date")
    findings: str = Field(..., description="Findings")
    impression: str = Field(..., description="Impression")
    tumorMeasurements: List[dict] = Field(..., description="Tumor measurements")

class RadiologySchema(BaseModel):
    imaging: List[Imaging] = Field(..., description="Imaging studies")

class LabResultsSchema(BaseModel):
    # Dynamic for lab results - LLM extracts all key-value pairs
    labs: dict = Field(..., description="All lab results as key-value pairs")

# Type to schema mapping
TYPE_TO_SCHEMA = {
    "Clinical Notes": ClinicalNotesSchema,
    "Pathology": PathologySchema,
    "Genomic Report": GenomicsSchema,
    "Radiology": RadiologySchema,
    "Lab Results": LabResultsSchema,
    "Other": ClinicalNotesSchema,  # Fallback to clinical notes schema
}

# Base prompt for structured extraction
STRUCTURED_PROMPT_TEMPLATE = """Extract information from the following document content into the exact JSON schema provided below. Fill in all fields with relevant data from the content. If data is missing, use null or empty strings/arrays where appropriate. Be precise and do not add extra fields.

Document content: {content}

SCHEMA: {schema}

Output ONLY the valid JSON matching this schema, no additional text."""

def get_structured_prompt(schema_class):
    parser = PydanticOutputParser(pydantic_object=schema_class)
    prompt = PromptTemplate(
        template=STRUCTURED_PROMPT_TEMPLATE,
        input_variables=["content"],
        partial_variables={"schema": parser.get_format_instructions()},
    )
    return prompt, parser

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

def process_file(temp_path: str, file_type: str, selected_model: str, is_vision: bool) -> Tuple[Dict, Dict, float]:
    """Process a single file with type-specific schema."""
    start_time = time.time()
    ext = Path(temp_path).suffix.lower()
    llm = get_llm(selected_model)
    schema_class = TYPE_TO_SCHEMA.get(file_type, ClinicalNotesSchema)  # Fallback
    prompt, parser = get_structured_prompt(schema_class)
    
    # Load content (text or vision)
    if ext == ".pdf":
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
        if text.strip() and not is_vision:
            # Text mode
            messages = prompt.format_messages(content=text)
            response = llm.invoke(messages)
            cleaned = clean_markdown(response.content)
            result = parser.parse(cleaned)
            usage_metadata = response.usage_metadata
        else:
            # Vision mode
            if not is_vision:
                raise ValueError("Scanned PDF requires vision model.")
            doc = fitz.open(temp_path)
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
            
            vision_prompt = PromptTemplate(
                template=STRUCTURED_PROMPT_TEMPLATE + "\n\nSCHEMA: {schema}",
                input_variables=[],
                partial_variables={
                    "schema": parser.get_format_instructions(),
                    "content": "Analyze the images below as document pages."
                },
            )
            message = HumanMessage(
                content=[{"type": "text", "text": vision_prompt.format()}] + image_contents
            )
            response = llm.invoke([message])
            cleaned = clean_markdown(response.content)
            result = parser.parse(cleaned)
            usage_metadata = response.usage_metadata
    else:
        # Image
        if selected_model not in VISION_MODELS:
            raise ValueError("Image requires vision model.")
        with open(temp_path, "rb") as image_file:
            img_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        mime_type, _ = guess_type(temp_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("Not a supported image.")
        
        vision_prompt = PromptTemplate(
            template=STRUCTURED_PROMPT_TEMPLATE + "\n\nSCHEMA: {schema}",
            input_variables=[],
            partial_variables={
                "schema": parser.get_format_instructions(),
                "content": "Analyze the image below as a document."
            },
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt.format()},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_data}"}},
            ],
        )
        response = llm.invoke([message])
        cleaned = clean_markdown(response.content)
        result = parser.parse(cleaned)
        usage_metadata = response.usage_metadata
    
    latency = time.time() - start_time
    return result.dict(), usage_metadata, latency  # Convert to dict for JSON

def merge_results(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(left)
    merged.update(right)
    return merged

def keep_left(left, right):
    return left

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    results: Annotated[Dict[str, Dict[str, Any]], merge_results]
    temp_paths: Annotated[Dict[str, str], keep_left]
    model: Annotated[str, keep_left]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    results: Annotated[Dict[str, Dict[str, Any]], merge_results]
    temp_paths: Dict[str, str]
    model: str

# Start node
def start_node(state: AgentState, files_data: List[Tuple[str, str]]) -> AgentState:
    """Start: Prepare parallel processing."""
    state['files_data'] = files_data  # (file_name, type)
    return state

# Parallel extraction node (called dynamically per file)
def extract_node(state: AgentState, file_info: Tuple[str, str]) -> AgentState:
    file_name, file_type = file_info
    temp_path = state["temp_paths"][file_name]
    selected_model = state.get("model", "gpt-4o")
    is_vision = selected_model in VISION_MODELS

    try:
        result, usage_metadata, latency = process_file(
            temp_path, file_type, selected_model, is_vision
        )

        usd, inr = calculate_cost(usage_metadata, selected_model)

        metadata = {
            "input_tokens": usage_metadata.get("input_tokens", 0),
            "output_tokens": usage_metadata.get("output_tokens", 0),
            "total_tokens": usage_metadata.get("input_tokens", 0)
                            + usage_metadata.get("output_tokens", 0),
            "latency": latency,
            "cost_usd": usd,
            "cost_inr": inr,
        }

        return {
            "results": {
                file_name: {
                    "result": result,
                    "metadata": metadata,
                }
            }
        }

    except Exception as e:
        return {
            "results": {
                file_name: {
                    "error": str(e)
                }
            }
        }


# End node
def end_node(state: AgentState) -> AgentState:
    return state

def build_graph(files_data: List[Tuple[str, str]], temp_paths: Dict[str, str], selected_model: str):
    workflow = StateGraph(AgentState)

    workflow.add_node("start", lambda state: start_node(state, files_data))

    for i, (file_name, _) in enumerate(files_data):
        workflow.add_node(
            f"extract_{i}",
            lambda state, idx=i: extract_node(state, files_data[idx])
        )

    workflow.set_entry_point("start")

    for i in range(len(files_data)):
        workflow.add_edge("start", f"extract_{i}")
        workflow.add_edge(f"extract_{i}", END)

    app = workflow.compile()

    initial_state = {
        "messages": [],
        "results": {},
        "temp_paths": temp_paths,
        "model": selected_model
    }

    final_state = app.invoke(initial_state)
    return final_state["results"]


def main():
    # CLI unchanged (single file, dynamic)
    parser = argparse.ArgumentParser(description="Extract structured JSON from PDF or image using LangChain + OpenAI.")
    parser.add_argument("file_path", type=str, help="Path to the input PDF or image file.")
    parser.add_argument("--output", "-o", type=str, default="extracted.json", help="Output JSON file path (default: extracted.json).")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o", choices=list(MODEL_PRICING.keys()), help="Model name (default: gpt-4o).")
    parser.add_argument("--type", "-t", type=str, default="Other", choices=FILE_TYPES, help="File type (default: Other).")
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    llm = get_llm(args.model)
    is_vision = args.model in VISION_MODELS
    result, usage_metadata, latency = process_file(str(file_path), args.type, args.model, is_vision)
    
    usd, inr = calculate_cost(usage_metadata, args.model)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Extraction complete! Results saved to {args.output}")
    print(json.dumps(result, indent=2))
    input_t = usage_metadata.get('input_tokens', 0)
    output_t = usage_metadata.get('output_tokens', 0)
    total_t = input_t + output_t
    print(f"Input tokens: {input_t:,} | Output tokens: {output_t:,} | Total tokens: {total_t:,} tokens / ${usd:.6f} / ₹{inr:.2f} | Latency: {latency:.2f}s")

# Enhanced Streamlit UI for multi-file
# @st.cache_data
def save_uploaded_files(uploaded_files):
    temp_paths = {}
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_paths[uploaded_file.name] = tmp.name
    return temp_paths

def run_streamlit():
    st.title("🧠 Multi-Document GPT Extraction Tester")
    st.markdown("Upload multiple documents, assign types, and extract in parallel using LangGraph.")
    
    # Model selection
    models = list(MODEL_PRICING.keys())
    selected_model = st.selectbox("Select Model", models, index=models.index('gpt-4o'))
    is_vision = selected_model in VISION_MODELS
    
    # Multi-file upload
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF/Image)", 
        type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "webp"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Type selection for each file
        file_types = {}
        temp_paths = save_uploaded_files(uploaded_files)
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{uploaded_file.name}**")
            with col2:
                file_types[uploaded_file.name] = st.selectbox(
                    f"Type for {uploaded_file.name}",
                    FILE_TYPES,
                    key=f"type_{uploaded_file.name}",
                    index=FILE_TYPES.index("Other")
                )
        
        files_data = [(name, file_types[name]) for name in file_types]
        
        if st.button("Extract All in Parallel", type="primary"):
            with st.spinner(f"Processing {len(uploaded_files)} files with {selected_model}..."):
                try:
                    results = build_graph(files_data, temp_paths, selected_model)
                    
                    # Display per file
                    for file_name, data in results.items():
                        if 'error' in data:
                            st.error(f"Error for {file_name}: {data['error']}")
                            continue
                        
                        st.subheader(f"Results for {file_name} ({file_types[file_name]})")
                        
                        # JSON
                        st.json(data['result'])
                        
                        # Download
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(data['result'], indent=2),
                            file_name=f"{file_name}_{selected_model}.json",
                            key=f"dl_{file_name}"
                        )
                        
                        # Metrics in two columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            input_t = data['metadata']['input_tokens']
                            st.metric("Input Tokens", f"{input_t:,}")
                            output_t = data['metadata']['output_tokens']
                            st.metric("Output Tokens", f"{output_t:,}")
                        with col2:
                            total_t = data['metadata']['total_tokens']
                            st.metric("Total Tokens", f"{total_t:,}")
                            latency = data['metadata']['latency']
                            st.metric("Latency (s)", f"{latency:.2f}")
                        with col3:
                            usd = data['metadata']['cost_usd']
                            st.metric("Cost USD", f"${usd:.8f}")
                            inr = data['metadata']['cost_inr']
                            st.metric("Cost INR", f"₹{inr:.2f}")
                    
                    # Cleanup
                    for path in temp_paths.values():
                        os.unlink(path)
                    
                    st.success("All extractions complete!")
                    st.caption(f"Traced in LangSmith project '{os.getenv('LANGCHAIN_PROJECT')}'.")
                
                except Exception as e:
                    st.error(f"Parallel extraction failed: {e}")
    
    # Cleanup on session end (optional)
    # Note: Temp files cleaned after processing

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != "run":
        main()
    else:
        run_streamlit()

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

from pydantic import BaseModel, Field
import fitz
import base64
from mimetypes import guess_type

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage

# --------------------------------------------------
# ENV
# --------------------------------------------------

load_dotenv(override=True)

MODEL_NAME = "gpt-5-mini"
USD_TO_INR = 90

# --------------------------------------------------
# SCHEMAS
# --------------------------------------------------

class TreatmentHistory(BaseModel):
    id: str
    type: str
    name: str
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    status: Optional[str] = None
    response: Optional[str] = None
    responseDescription: Optional[str] = None
    notes: Optional[str] = None


class FamilyHistory(BaseModel):
    relation: str
    cancerType: str
    ageAtDiagnosis: Optional[int] = None
    deceased: Optional[bool] = None
    ageAtDeath: Optional[int] = None


class Diagnosis(BaseModel):
    primary: str
    icdCode: Optional[str] = None
    tnmStaging: dict
    dateOfDiagnosis: Optional[str] = None
    confirmedBy: Optional[str] = None
    method: Optional[str] = None
    notes: Optional[str] = None


class ClinicalNotesSchema(BaseModel):
    treatmentHistory: List[TreatmentHistory]
    familyHistory: List[FamilyHistory]
    diagnosis: Diagnosis


class PathologySpecimen(BaseModel):
    specimenType: str
    collectionDate: Optional[str] = None
    histologicalType: str
    grade: Optional[str] = None
    margins: Optional[str] = None
    lymphNodes: Optional[str] = None
    tumorSize: Optional[str] = None
    findings: List[str]
    ihcMarkers: List[dict]


class PathologySchema(BaseModel):
    pathology: List[PathologySpecimen]


class Imaging(BaseModel):
    id: str
    modality: str
    bodyRegion: str
    date: Optional[str] = None
    findings: str
    impression: Optional[str] = None
    tumorMeasurements: List[dict]


class RadiologySchema(BaseModel):
    imaging: List[Imaging]


class Biomarker(BaseModel):
    name: str
    category: str
    result: str
    status: str
    actionable: Optional[bool] = None
    clinicalRelevance: Optional[str] = None
    therapeuticImplications: List[str]
    testDate: Optional[str] = None
    methodology: Optional[str] = None


class GenomicsSchema(BaseModel):
    biomarkers: List[Biomarker]


class LabResultsSchema(BaseModel):
    labs: dict


# --------------------------------------------------
# TYPE → SCHEMA
# --------------------------------------------------

TYPE_TO_SCHEMA = {
    "Clinical Notes": ClinicalNotesSchema,
    "Pathology": PathologySchema,
    "Radiology": RadiologySchema,
    "Genomic Report": GenomicsSchema,
    "Lab Results": LabResultsSchema,
    "Other": ClinicalNotesSchema,
}

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

STRUCTURED_PROMPT = """
Extract information from the document into the JSON schema below.
If a value is missing, use null.
Do not add extra fields.

CONTENT:
{content}

SCHEMA:
{schema}

Return ONLY valid JSON.
"""

# --------------------------------------------------
# LLM + COST
# --------------------------------------------------

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        # temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def calculate_cost(usage):
    if not usage:
        return 0, 0, 0, 0
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    usd = (inp * 0.25 + out * 2.00) / 1_000_000
    inr = usd * USD_TO_INR
    return inp, out, inp + out, usd, inr


# --------------------------------------------------
# CORE EXTRACTION
# --------------------------------------------------
def is_scanned_pdf(text: str) -> bool:
    return not text or len(text.strip()) < 50

def process_file(path: str, doc_type: str):
    start = time.time()
    llm = get_llm()

    schema_cls = TYPE_TO_SCHEMA[doc_type]
    parser = PydanticOutputParser(pydantic_object=schema_cls)

    ext = Path(path).suffix.lower()
    response = None  # <-- IMPORTANT: ensure defined

    # --------------------------------------------------
    # PDF HANDLING
    # --------------------------------------------------
    if ext == ".pdf":
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        text = "\n\n".join(d.page_content for d in docs)

        # ---------- TEXT PDF ----------
        if not is_scanned_pdf(text):
            prompt = ChatPromptTemplate.from_template(STRUCTURED_PROMPT)
            messages = prompt.format_messages(
                content=text,
                schema=parser.get_format_instructions()
            )
            response = llm.invoke(messages)

        # ---------- SCANNED PDF (VISION) ----------
        else:
            doc = fitz.open(path)
            image_messages = []

            for page in doc:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode()

                image_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                })

            doc.close()

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": STRUCTURED_PROMPT.format(
                            content="Analyze the document pages provided as images.",
                            schema=parser.get_format_instructions()
                        )
                    },
                    *image_messages
                ]
            )

            response = llm.invoke([message])

    # --------------------------------------------------
    # IMAGE HANDLING (JPG / PNG / etc.)
    # --------------------------------------------------
    else:
        mime_type, _ = guess_type(path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Unsupported file type: {path}")

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": STRUCTURED_PROMPT.format(
                        content="Analyze the image below as a medical document.",
                        schema=parser.get_format_instructions()
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_b64}"
                    }
                }
            ]
        )

        response = llm.invoke([message])

    # --------------------------------------------------
    # SAFETY CHECK (IMPORTANT)
    # --------------------------------------------------
    if response is None:
        raise RuntimeError("LLM response was not generated")

    result = parser.parse(response.content)
    latency = time.time() - start

    return result.dict(), response.usage_metadata, latency


# --------------------------------------------------
# LANGGRAPH
# --------------------------------------------------

def merge_results(a, b):
    x = dict(a)
    x.update(b)
    return x


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    results: Annotated[Dict[str, Any], merge_results]
    temp_paths: Dict[str, str]


def extract_node(state: AgentState, info: Tuple[str, str]) -> AgentState:
    fname, dtype = info
    path = state["temp_paths"][fname]

    try:
        text, usage, latency = process_file(path, dtype)
        inp, out, total, usd, inr = calculate_cost(usage)

        return {
            "results": {
                fname: {
                    "text": text,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "total_tokens": total,
                    "cost_usd": usd,
                    "cost_inr": inr,
                    "latency": latency
                }
            }
        }
    except Exception as e:
        return {"results": {fname: {"error": str(e)}}}


def build_graph(files_data, temp_paths):
    results = {}

    for file_name, doc_type in files_data:
        try:
            result, usage, latency = process_file(
                temp_paths[file_name],
                doc_type
            )

            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)

            results[file_name] = {
                "text": result,
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": input_t + output_t,
                "cost_usd": (input_t * 0.25 + output_t * 2.0) / 1_000_000,
                "cost_inr": ((input_t * 0.25 + output_t * 2.0) / 1_000_000) * 90,
                "latency": latency
            }

        except Exception as e:
            results[file_name] = {"error": str(e)}

    return results


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(title="Multi-Document Extraction API")


@app.post("/extract")
async def extract_api(
    files: List[UploadFile] = File(...),
    types: List[str] = Form(...)
):
    if len(files) != len(types):
        return JSONResponse(
            status_code=400,
            content={"error": "files and types count mismatch"}
        )

    temp_paths = {}
    files_data = []

    for f, t in zip(files, types):
        suffix = Path(f.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await f.read())
            temp_paths[f.filename] = tmp.name
            files_data.append((f.filename, t))

    results = build_graph(files_data, temp_paths)

    for p in temp_paths.values():
        os.unlink(p)

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )

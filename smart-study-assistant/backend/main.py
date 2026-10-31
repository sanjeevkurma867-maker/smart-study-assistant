import os, json, uuid
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai

load_dotenv()
app = FastAPI(title="Smart Study Assistant API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key) if key else None
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
documents = {}

class StudyRequest(BaseModel):
    document_id: str
    question: Optional[str] = None
    difficulty: Optional[str] = "medium"
    number_of_questions: Optional[int] = 5

def ai(prompt):
    if not client:
        raise HTTPException(503, "Gemini API key missing. Add GEMINI_API_KEY to backend/.env.")
    try:
        return client.models.generate_content(model=MODEL, contents=prompt).text or "No response generated."
    except Exception as e:
        raise HTTPException(502, f"AI request failed: {e}")

def get_doc(doc_id):
    text = documents.get(doc_id)
    if not text:
        raise HTTPException(404, "Document not found. Please upload it again.")
    return text

@app.get("/")
def root():
    return {"message":"Smart Study Assistant API is running", "docs":"/docs"}

@app.get("/api/health")
def health():
    return {"status":"ok", "gemini_configured": client is not None}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "notes"
    if not (name.lower().endswith(".pdf") or name.lower().endswith(".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported.")
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Maximum file size is 12 MB.")
    try:
        if name.lower().endswith(".pdf"):
            import io
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            text = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    text = text.strip()
    if not text:
        raise HTTPException(400, "No readable text found. Scanned PDFs need OCR.")
    text = text[:100000]
    doc_id = str(uuid.uuid4())
    documents[doc_id] = text
    return {"document_id":doc_id, "filename":name, "characters_extracted":len(text), "message":"Upload successful."}

@app.post("/api/ask")
def ask(req: StudyRequest):
    text = get_doc(req.document_id)
    if not (req.question or "").strip():
        raise HTTPException(400, "Enter a question.")
    prompt = f"""You are a friendly study assistant. Answer using the study material. If the answer is absent, say so clearly. Explain in student-friendly language.
STUDY MATERIAL:
{text[:70000]}
QUESTION:
{req.question}"""
    return {"answer":ai(prompt)}

@app.post("/api/summarize")
def summarize(req: StudyRequest):
    text = get_doc(req.document_id)
    prompt = f"""Create revision notes from this material. Include an overview, key concepts and definitions, bullet-point notes, formulas/examples if present, and a short revision checklist. Do not invent source details.
MATERIAL:
{text[:70000]}"""
    return {"summary":ai(prompt)}

@app.post("/api/quiz")
def quiz(req: StudyRequest):
    text = get_doc(req.document_id)
    count = max(3, min(req.number_of_questions or 5, 15))
    prompt = f"""Create {count} multiple-choice questions based only on the material. Difficulty: {req.difficulty or 'medium'}. Include four options A-D, correct answer and one-sentence explanation. Return valid JSON only in this format:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","explanation":"..."}}]}}
MATERIAL:
{text[:70000]}"""
    raw = ai(prompt).strip()
    if raw.startswith("```"):
        raw = raw.split("\n",1)[-1].removesuffix("```").strip()
    try:
        data = json.loads(raw)
        if not isinstance(data.get("questions"), list): raise ValueError()
        return data
    except Exception:
        return {"raw_quiz":raw, "note":"AI returned non-JSON quiz text."}

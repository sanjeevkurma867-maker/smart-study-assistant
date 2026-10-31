# Smart Study Assistant

A full-stack study helper with document upload, AI Q&A, summaries and practice quizzes.

## Stack
- React + Vite frontend
- Python + FastAPI backend
- Gemini API for AI generation
- pypdf for extracting PDF text

## Setup

### Backend
```bash
cd smart-study-assistant/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Open `backend/.env`, add your Gemini API key, then run:
```bash
uvicorn main:app --reload
```
API: http://localhost:8000 | Docs: http://localhost:8000/docs

### Frontend (open a second terminal)
```bash
cd smart-study-assistant/frontend
npm install
npm run dev
```
Open http://localhost:5173

## Use
1. Upload a PDF or TXT file (max 12 MB).
2. Ask questions about the uploaded notes.
3. Generate a revision summary.
4. Generate a multiple-choice quiz and reveal answers.

## Notes
- Get a Gemini API key from Google AI Studio. Keep it only in `backend/.env`; never put it in frontend code or commit it.
- Uploaded document text is held in backend memory and is cleared when the backend restarts.
- Scanned/image-only PDFs are not supported (OCR is not included).
- Extracted document text is sent to Gemini when you use AI features. Do not upload confidential documents unless comfortable with that.
- This is a starter/academic project. For production, add authentication, persistent storage, chunking + vector retrieval, rate limits and stronger file validation.

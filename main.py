import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from app.schemas import ChatMessage, ChatResponse
from app.retriever import get_rag_response

# -----------------------------
# Rate Limiter Setup
# -----------------------------
limiter = Limiter(key_func=get_remote_address)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="DocuMind Enterprise API",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

# -----------------------------
# Static Files (Chat UI)
# -----------------------------
import pathlib
static_dir = pathlib.Path("app/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# -----------------------------
# Root Route — Serves Chat UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <head><title>DocuMind Enterprise</title></head>
        <body style='background:#1a1a2e;color:white;font-family:Arial;text-align:center;padding:50px'>
            <h1 style='color:#e94560'>DocuMind Enterprise</h1>
            <p>index.html not found in app/static/</p>
            <p>Please create app/static/index.html</p>
            <br>
            <a href='/docs' style='color:#e94560'>Go to API Docs</a>
        </body>
        </html>
        """

# -----------------------------
# Health Check Route
# -----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "DocuMind Enterprise"
    }

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatMessage):
    try:
        result = get_rag_response(
            body.question,
            body.chat_history
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -----------------------------
# PDF Upload & Auto-Index Endpoint
# -----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )
    try:
        path = f"docs/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        from app.ingest import load_documents, chunk_documents, embed_and_upsert
        docs = load_documents("./docs")
        chunks = chunk_documents(docs)
        embed_and_upsert(chunks)

        return {
            "message": f"{file.filename} uploaded and indexed successfully",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -----------------------------
# List Indexed Documents
# -----------------------------
@app.get("/documents")
async def list_documents():
    try:
        docs = []
        for file in os.listdir("./docs"):
            if file.endswith(".pdf"):
                size = os.path.getsize(f"./docs/{file}")
                docs.append({
                    "filename": file,
                    "size_kb": round(size / 1024, 2)
                })
        return {
            "total": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

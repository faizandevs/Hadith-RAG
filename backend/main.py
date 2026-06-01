# main.py — FastAPI app
# run locally: uvicorn main:app --reload
# deploy: Render reads this file via start command

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import rag

# ── startup/shutdown ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once on startup — loads model + ChromaDB
    rag.load_resources()
    yield
    # runs on shutdown — nothing to clean up

app = FastAPI(
    title="Bukhari RAG API",
    description="RAG chatbot over Sahih Bukhari",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow frontend to call this API ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your GitHub Pages URL in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── request/response models ───────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer    : str
    citations : list[str]
    chunks    : list[str]
    query     : str

# ── endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    """Render uses this to check if app is alive"""
    return {"status": "ok", "hadiths_indexed": rag._collection.count()}

@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(req.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars)")
    
    result = rag.answer_query(req.query)
    return result

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
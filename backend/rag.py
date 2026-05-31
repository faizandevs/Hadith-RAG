# rag.py — all RAG logic, imported by main.py
# keeps main.py clean and this testable independently

import os
import re
import time
import numpy as np
import chromadb
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
from dotenv import load_dotenv
load_dotenv() 
# ── config (loaded from environment) ─────────────────────────
CHROMA_DIR        = os.environ.get("CHROMA_DIR", "./chroma_store/chroma_store")
EMBED_MODEL       = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
OPENROUTER_KEY    = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL         = "openrouter/free"
COLLECTION_NAME   = "hadith_aware"
TOP_K             = 3
FETCH_K           = 25

# ── globals (loaded once on startup) ─────────────────────────
_embed_model  : Optional[SentenceTransformer] = None
_collection   : Optional[chromadb.Collection] = None

class LocalEF(chromadb.EmbeddingFunction):
    def __init__(self, model):
        self.model = model
    def __call__(self, input):
        return self.model.encode(
            input, batch_size=64, show_progress_bar=False
        ).tolist()

def load_resources():
    """called once on FastAPI startup — loads model + ChromaDB"""
    global _embed_model, _collection
    
    print(f"Loading embedding model: {EMBED_MODEL}")
    _embed_model = SentenceTransformer(EMBED_MODEL)
    
    print(f"Loading ChromaDB from: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalEF(_embed_model)
    )
    count = _collection.count()
    print(f"✓ Collection loaded — {count} hadiths indexed")


# ── MMR retrieval ─────────────────────────────────────────────
def mmr_retrieve(
    query: str,
    k: int = TOP_K,
    fetch_k: int = FETCH_K,
    lambda_val: float = 0.6,
) -> Tuple[List[str], List[float]]:
    
    resp = _collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "distances", "embeddings"],
    )
    candidates     = resp["documents"][0]
    query_sims     = [1 - d for d in resp["distances"][0]]
    candidate_embs = np.array(resp["embeddings"][0])

    def cosine(a, b):
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        )

    selected, remaining = [], list(range(len(candidates)))
    while len(selected) < k and remaining:
        scores = {}
        for i in remaining:
            relevance  = query_sims[i]
            redundancy = max(
                (cosine(candidate_embs[i], candidate_embs[j]) for j in selected),
                default=0.0
            )
            scores[i] = lambda_val * relevance - (1 - lambda_val) * redundancy
        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.remove(best)

    return (
        [candidates[i] for i in selected],
        [query_sims[i]  for i in selected],
    )


# ── LLM call with retry ───────────────────────────────────────
def llm_call(messages: list, max_tokens: int = 800) -> Optional[str]:
    for attempt in range(3):
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type"  : "application/json",
                "Authorization" : f"Bearer {OPENROUTER_KEY}",
                "HTTP-Referer"  : "https://bukhari-rag.onrender.com",
                "X-Title"       : "Bukhari RAG",
            },
            json={
                "model"     : LLM_MODEL,
                "messages"  : messages,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        raw = resp.json()

        if raw.get("error", {}).get("code") == 429:
            wait = raw["error"].get(
                "metadata", {}
            ).get("retry_after_seconds", 15)
            time.sleep(wait + 1)
            continue

        if "choices" not in raw:
            return None

        msg = raw["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning") or None

    return None


# ── HyDE retrieval ────────────────────────────────────────────
# in rag.py — replace hyde_retrieve function
def hyde_retrieve(query: str) -> Tuple[List[str], List[float], str]:
    
    # force the model to skip reasoning and output directly
    hyde_prompt = f"""Narrated Umar bin Al-Khattab: The Prophet said regarding {query.lower().rstrip('?')}: [complete this hadith in 2 sentences, output ONLY the hadith text, no explanation]"""
    
    hypothetical = llm_call(
        messages=[
            {"role": "user", "content": hyde_prompt}
        ],
        max_tokens=150,
    )
    
    # strip reasoning artifacts if model still thinks out loud
    # reasoning models often put the real answer after the last newline
    if hypothetical:
        lines = [l.strip() for l in hypothetical.strip().split('\n') if l.strip()]
        # take last 2 lines — usually where the actual answer lands
        hypothetical = ' '.join(lines[-2:]) if len(lines) > 2 else hypothetical
    
    if not hypothetical:
        hypothetical = query
    
    print(f"HyDE hypothetical (cleaned): {hypothetical[:200]}")
    chunks, sims = mmr_retrieve(hypothetical, k=TOP_K, fetch_k=FETCH_K)
    return chunks, sims, hypothetical


# ── main answer function ──────────────────────────────────────
def answer_query(query: str) -> dict:
    """
    Full RAG pipeline:
    query → HyDE → MMR retrieve → LLM answer
    Returns dict with answer + source citations
    """
    chunks, sims, hypothetical = hyde_retrieve(query)

    # extract hadith references from chunks for citations
    citations = []
    for chunk in chunks:
        match = re.search(
            r"Volume\s+\d+,\s+Book\s+\d+,\s+Number\s+\d+", chunk
        )
        if match:
            citations.append(match.group(0))

    context = "\n\n---\n\n".join(chunks)

    answer = llm_call(
        messages=[{"role": "user", "content": f"""You are a knowledgeable assistant for Sahih Bukhari.
Use ONLY the provided hadith excerpts to answer the question.
If the answer is not in the excerpts, say "I could not find this in the provided hadiths."
Always mention the Volume, Book and Number of the hadith you are citing.
Be concise and respectful in tone.

Hadith excerpts:
{context}

Question: {query}

Answer:"""}],
        max_tokens=500,
    )

    return {
        "answer"    : answer or "Could not generate answer. Please try again.",
        "citations" : citations,
        "chunks"    : chunks,         # full text for frontend to show
        "query"     : query,
    }
# Bukhari RAG — Hadith Chatbot

A production RAG (Retrieval-Augmented Generation) system over Sahih Bukhari,
built as a deep dive into chunking strategies and retrieval quality.

## Architecture

User query
→ HyDE (LLM generates hypothetical answer)
→ MMR retrieval from ChromaDB (7,128 hadiths)
→ LLM generates grounded answer with citations
→ Frontend displays answer + source hadiths

## What I learned building this

- Compared 5 chunking strategies on a real 1,700-page Arabic-origin PDF
- Hadith-aware chunking (one chunk = one hadith) outperformed generic fixed-size
- Standard top-k retrieval returned near-duplicate hadiths — MMR fixed this
- HyDE bridged the semantic gap between user questions and hadith language
- Evaluated with hit@3 and MRR metrics across 12 test queries

See `notebook/rag_deep_dive.ipynb` for the full research and eval results.

## Tech stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, local)
- **Vector DB**: ChromaDB (persistent, local)
- **Retrieval**: MMR + HyDE
- **LLM**: OpenRouter free tier
- **Backend**: FastAPI on Render
- **Frontend**: Vanilla HTML/CSS/JS on GitHub Pages

## Local setup

```bash
git clone https://github.com/yourusername/bukhari-rag
cd bukhari-rag/backend

python -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate

pip install -r requirements.txt

# add your key
cp .env.example .env
# edit .env and add OPENROUTER_API_KEY=your-key

# copy your chroma_store into backend/
# then run
uvicorn main:app --reload
```

Frontend: open `frontend/index.html` in browser — change `API_URL` to `http://localhost:8000` for local dev.

## Deployment

- Backend: Render (free tier) — connect GitHub repo, set env vars, start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Frontend: GitHub Pages — enable in repo Settings → Pages → main branch /frontend folder

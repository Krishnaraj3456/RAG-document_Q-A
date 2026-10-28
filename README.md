# RAG Document Q&A Service

Production-style retrieval-augmented generation over a PDF report.

## Architecture
PDF → clean_text (drop TOC/page numbers) → split_by_sections (header-aware)
→ all-MiniLM-L6-v2 embeddings → ChromaDB
→ vector top-20 → cross-encoder rerank → top-3
→ Groq LLM with grounded prompt + refusal guardrail

## Results
- MRR 1.0 on a 4-question golden set
- Refusal verified on out-of-corpus questions
- Section-level citations on every answer

## Stack
FastAPI · ChromaDB · sentence-transformers · cross-encoder · Groq · Docker

## Run locally
1. `python -m venv venv && .\venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Add `GROQ_API_KEY=...` to `.env`
4. `python ingest.py`
5. `uvicorn api:app --reload`
6. Open http://127.0.0.1:8000/docs

## What I learned
- Density filters are backwards for prose vs tables
- Section-aware chunking beats character splitting on real documents
- Cross-encoder reranking lifts MRR by ~0.25 on this corpus
- Guardrails must be structural — one entry point, always retrieve
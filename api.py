import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
from rerank_search import search_with_rerank

load_dotenv()
app = FastAPI(title="RAG Document Q&A")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer ONLY using the provided context.
- If the answer is not in the context, respond exactly: "I could not find that in the document."
- Cite the section for every claim using square brackets only, like this: [Section: 8.3 Limitations]
- Do not use 【】 or any other bracket style.
- Be concise. Do not use outside knowledge."""

class Question(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    results = search_with_rerank(q.text, top_k=3, candidate_k=20)
    context = "\n\n".join(
        f"[Section: {r['metadata'].get('section', 'unknown')}]\n{r['document']}"
        for r in results
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {q.text}"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            timeout=20,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "answer": resp.choices[0].message.content,
        "citations": [
            {"section": r["metadata"].get("section"),
             "rerank_score": float(round(r["rerank_score"], 3))}
            for r in results
        ],
    }
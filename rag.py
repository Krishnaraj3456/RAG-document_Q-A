import os
from dotenv import load_dotenv
from groq import Groq
from rerank_search import search_with_rerank

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer ONLY using the provided context.
- If the answer is not in the context, respond exactly: "I could not find that in the document."
- Cite the section for every claim, like: [Section: 5.2 Minimum Support Threshold].
- Be concise. Do not use outside knowledge."""


def answer(question: str, top_k: int = 3):
    results = search_with_rerank(question, top_k=top_k, candidate_k=20)

    context_blocks = []
    for i, r in enumerate(results, start=1):
        section = r["metadata"].get("section", "unknown")
        context_blocks.append(f"[Chunk {i} | Section: {section}]\n{r['document']}")
    context = "\n\n".join(context_blocks)

    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        timeout=20,
    )
    answer_text = response.choices[0].message.content

    return {
        "answer": answer_text,
        "citations": [
            {"section": r["metadata"].get("section"), "rerank_score": float(round(r["rerank_score"], 3))}
            for r in results
        ],
    }


if __name__ == "__main__":
    tests = [
        "What dataset was used and how many transactions?",
        "What is the minimum support threshold?",
        "What is the Apriori algorithm?",          # NOT in the PDF — must refuse
        "Which region has the highest profit margin?",
        "What is the minimum support threshold?",              # in PDF — should answer
        "What is the revenue of Amazon in 2024?",             # NOT in PDF — should refuse
        "Who won the 2022 FIFA World Cup?",                    # NOT in PDF — should refuse
        "What is the capital of France?",                      # NOT in PDF — should refuse
    ]
    for q in tests:
        print(f"\n=== {q} ===")
        out = answer(q)
        print(out["answer"])
        print("Citations:", out["citations"])
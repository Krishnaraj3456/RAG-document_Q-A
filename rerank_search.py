import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Bi-encoder for initial retrieval (fast, approximate)
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-encoder for reranking (slow, accurate)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("docs")


def search_with_rerank(query: str, top_k: int = 5, candidate_k: int = 20):
    # Stage 1: vector search — get 20 candidates
    q_emb = bi_encoder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=candidate_k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # Stage 2: cross-encoder reranks each (query, doc) pair
    pairs = [(query, doc) for doc in docs]
    rerank_scores = cross_encoder.predict(pairs)

    # Stage 3: sort by rerank score
    ranked = sorted(
        zip(docs, metas, distances, rerank_scores),
        key=lambda x: x[3],
        reverse=True,
    )[:top_k]

    return [
        {"document": d, "metadata": m, "distance": dist, "rerank_score": score}
        for d, m, dist, score in ranked
    ]
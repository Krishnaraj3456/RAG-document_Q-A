from rerank_search import search_with_rerank

GOLDEN = [
    {"q": "What dataset was used and how many transactions?", "expected_keyword": "3,000+ transactions"},
    {"q": "What is the minimum support threshold?", "expected_keyword": "support threshold (2%)"},
    {"q": "Which algorithms were compared in the study?", "expected_keyword": "Apriori, FP-Growth"},
    {"q": "Which region has the highest profit margin?", "expected_keyword": "Central shows the highest profit margin"},
]

mrr_sum = 0
for item in GOLDEN:
    results = search_with_rerank(item["q"], top_k=5, candidate_k=20)
    top_docs = [r["document"] for r in results]

    rank = None
    for i, d in enumerate(top_docs, start=1):
        if item["expected_keyword"].lower() in d.lower():
            rank = i
            break

    if rank:
        mrr_sum += 1 / rank

    print(f"\nQ: {item['q']}")
    print(f"   Keyword {item['expected_keyword']!r} → rank={rank}")
    for i, r in enumerate(results, start=1):
        marker = " <-- MATCH" if item["expected_keyword"].lower() in r["document"].lower() else ""
        print(f"   [{i}] rerank={r['rerank_score']:.3f} | vec_dist={r['distance']:.3f} | {r['document'][:150]!r}{marker}")

print(f"\nMRR = {mrr_sum / len(GOLDEN):.3f}")
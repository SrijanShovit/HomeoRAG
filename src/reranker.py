from sentence_transformers import CrossEncoder
import time

from src.config import RERANKER_MODEL,K_RERANKING


_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker



def rerank(query: str, results: list, top_k: int = K_RERANKING):
    """
    results = output from semantic_search()["results"]
    Each item must have: text, medicine, synonyms, score
    """

    reranker = get_reranker()

    pairs = [(query, r["text"]) for r in results]

    start = time.perf_counter()
    scores = reranker.predict(pairs)
    latency_ms = (time.perf_counter() - start) * 1000

    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return {
        "latency_ms": round(latency_ms, 2),
        "results": results[:top_k]
    }


from src.rrf_retrieval.hybrid_search import rrf_retrieve


def retrieve_with_rrf(query, reranking_k):
    docs = rrf_retrieve(query)
    return docs[:reranking_k]

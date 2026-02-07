from src.rrf_retrieval.hybrid_search import rrf_retrieve


def retrieve_with_rrf(query, loaded_retrievers,reranking_k):
    docs = rrf_retrieve(query,loaded_retrievers)
    return docs[:reranking_k]

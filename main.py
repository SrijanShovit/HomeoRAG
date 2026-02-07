import asyncio
import os

# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.bm25_store import build_bm25_docs
from src.evaluate_retrieval import evaluate_query_set, evaluate_query_set_bm25,evaluate_query_set_rrf
from src.parser import split_medicines
from src.query_tester import query_phase_semantic,query_phase_keyword


if __name__ == "__main__":
    # index_phase()
    # query_phase_semantic()
    # evaluate_query_set(USE_RERANKING=True)

    evaluate_query_set_rrf(USE_RERANKING=True)

    # build_bm25_docs()
    # query_phase_keyword()
    # evaluate_query_set_bm25()

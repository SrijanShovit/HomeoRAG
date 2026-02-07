import asyncio
import os

# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.evaluate_retrieval import evaluate_query_set
from src.query_tester import query_phase
from src.vector_indexer import index_phase


if __name__ == "__main__":
    # index_phase()
    # query_phase()
    evaluate_query_set(USE_RERANKING=True)



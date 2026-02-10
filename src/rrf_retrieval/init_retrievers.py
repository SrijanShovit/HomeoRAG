# src/rrf_retrieval/init_retrievers.py

import time
from src.rrf_retrieval.all_retrievers import RETRIEVER_POOL
from src.config import ACTIVE_RRF_MODELS

def init_all_retrievers():
    loaded_retrievers = {}
    print("Initializing all RRF retrievers...")

    for name in ACTIVE_RRF_MODELS:

        retriever = RETRIEVER_POOL[name]
        print(f"  -> Loading {name}...", end=" ", flush=True)

        start = time.perf_counter()

        # run a dummy query to warm up models / indexes
        if name != "bm25_word_tokenize": #to skip tracing BM25Retriver on warm-up
            try:
                _  = retriever.search("kali", 1)
            except Exception:
                pass 

        elapsed = (time.perf_counter() - start) * 1000
        print(f"done ({elapsed:.1f} ms)")

        loaded_retrievers[name] = retriever

    print("All retrievers initialized")
    return loaded_retrievers

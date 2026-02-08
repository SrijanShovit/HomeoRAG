# src/rrf_retrieval/hybrid_search.py

import time
from src.rrf_retrieval.all_retrievers import RETRIEVER_POOL
from src.rrf_retrieval.rrf import reciprocal_rank_fusion
from src.config import ACTIVE_RRF_MODELS, RRF_RETRIEVAL_K


def rrf_retrieve(query,loaded_retrievers=RETRIEVER_POOL):
    model_results = {}

    print(f"\n[RRF] Query: {query}")

    # ----------------------------
    # Run all retrievers
    # ----------------------------
    for name in ACTIVE_RRF_MODELS:
        print(f"[RRF] Running {name} ...", end=" ", flush=True)
        retriever = loaded_retrievers[name]
        
        start = time.perf_counter()
        
        docs = retriever.search(query, RRF_RETRIEVAL_K)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Extra check for BM25
        # if "bm25" in name.lower():
        #     print(f"  -> BM25 retriever used. Requested k={RRF_RETRIEVAL_K}, returned {len(docs)} docs")
        
        # print(f"done ({len(docs)} docs, {elapsed:.1f} ms)")
        
        model_results[name] = docs

    # ----------------------------
    # Fusion
    # ----------------------------
    print("[RRF] Fusing results ...", end=" ", flush=True)

    start = time.perf_counter()
    fused = reciprocal_rank_fusion(model_results)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"done ({len(fused)} fused docs, {elapsed:.1f} ms)")

    return fused

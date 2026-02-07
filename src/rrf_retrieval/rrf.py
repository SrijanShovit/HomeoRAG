# src/rrf_retrieval/rrf.py

from collections import defaultdict
from src.config import RRF_K0


def reciprocal_rank_fusion(model_results):
    scores = defaultdict(float)
    doc_map = {}

    total_inputs = 0

    # ----------------------------
    # Collect RRF scores
    # ----------------------------
    for model, docs in model_results.items():
        total_inputs += len(docs)

        for rank, doc in enumerate(docs, start=1):
            scores[doc["id"]] += 1 / (RRF_K0 + rank)
            doc_map[doc["id"]] = doc

    # ----------------------------
    # Diagnostics
    # ----------------------------
    unique_docs = len(scores)

    # print(
    #     f"[RRF] Inputs: {total_inputs}  |  "
    #     f"Unique after merge: {unique_docs}  |  "
    #     f"Duplicates removed: {total_inputs - unique_docs}"
    # )

    # ----------------------------
    # Build fused list
    # ----------------------------
    fused = []
    for doc_id, score in scores.items():
        d = doc_map[doc_id].copy()
        d["rrf_score"] = score
        fused.append(d)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused

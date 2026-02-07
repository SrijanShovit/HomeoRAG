import json
import re
import numpy as np
from src.config import K_RETRIEVAL,K_RERANKING, TEST_QUERIES_FINAL_DATA
from src.keyword_search import BM25VectorDB, get_bm25
from src.reranker import rerank
from src.rrf_retrieval.retrieval_pipeline import retrieve_with_rrf
from src.semantic_search import semantic_search


# ------------------------------
# Data loading
# ------------------------------
def load_test_queries(path=TEST_QUERIES_FINAL_DATA):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------
# Section extraction
# ------------------------------
def extract_sections_from_chunk(text):
    pattern = r"([A-Z][a-zA-Z]+)\.\––"
    return set(re.findall(pattern, text))


# ------------------------------
# Ranking metrics
# ------------------------------
def calc_mrr(rows):
    rr = [1 / r["remedy_rank"] for r in rows if r["remedy_rank"] is not None]
    return np.mean(rr) if rr else 0


def graded_relevance(section_match, remedy_match):
    if section_match and remedy_match:
        return 3
    if section_match:
        return 2
    if remedy_match:
        return 1
    return 0


def dcg(scores):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(scores))


def ndcg(scores):
    ideal = sorted(scores, reverse=True)
    if sum(ideal) == 0:
        return 0.0
    return dcg(scores) / dcg(ideal)


# ------------------------------
# Document matching
# ------------------------------
def doc_matches(doc, expected_sections, best_remedy):
    doc_sections = extract_sections_from_chunk(doc["text"])
    section_match = bool(expected_sections & doc_sections)
    remedy_match = (doc["medicine"] == best_remedy)
    return section_match, remedy_match


# ------------------------------
# Per-query evaluation
# ------------------------------
def evaluate_query(reranked_results, test_item):
    expected_sections = set(test_item["expected_sections"])
    best_remedy = test_item["best_remedy"]

    section_hit = False
    remedy_hit = False
    remedy_rank = None
    relevance_scores = []

    for rank, doc in enumerate(reranked_results, start=1):
        section_match, remedy_match = doc_matches(
            doc, expected_sections, best_remedy
        )

        if section_match:
            section_hit = True

        if remedy_match and remedy_rank is None:
            remedy_hit = True
            remedy_rank = rank

        relevance_scores.append(
            graded_relevance(section_match, remedy_match)
        )

    return {
        "query_id": test_item["query_id"],
        f"section_hit@{K_RERANKING}": section_hit,
        f"remedy_hit@{K_RERANKING}": remedy_hit,
        "remedy_rank": remedy_rank,
        f"ndcg@{K_RERANKING}": ndcg(relevance_scores)
    }


# ------------------------------------------------------------
# Full benchmark One Embedding Model Semantic Search + Re-rank
# ------------------------------------------------------------
def evaluate_query_set(USE_RERANKING:bool=True):
    test_set = load_test_queries(path=TEST_QUERIES_FINAL_DATA)
    rows = []

    for item in test_set[:100]:
        question = item["query"]

        dense = semantic_search(question, k=K_RETRIEVAL)

        if USE_RERANKING:
            final_results = rerank(
                question,
                dense["results"],
                top_k=K_RERANKING
            )["results"]
        else:
            final_results = dense["results"][:K_RERANKING]

        result = evaluate_query(final_results, item)
        print(result)
        rows.append(result)

    metrics = {
        f"Recall@{K_RERANKING}": np.mean([r[f"section_hit@{K_RERANKING}"] for r in rows]),
        f"RemedyHit@{K_RERANKING}": np.mean([r[f"remedy_hit@{K_RERANKING}"] for r in rows]),
        "MRR": calc_mrr(rows),
        f"NDCG@{K_RERANKING}": np.mean([r[f"ndcg@{K_RERANKING}"] for r in rows]),
    }

    print("\nFINAL METRICS")
    print(metrics)
    return metrics



# ------------------------------------------------------------
# Full benchmark: RRF retrieval + optional re-ranking
# ------------------------------------------------------------
def evaluate_query_set_rrf(USE_RERANKING: bool = True):
    test_set = load_test_queries(path=TEST_QUERIES_FINAL_DATA)
    rows = []

    for item in test_set[:100]:
        query = item["query"]

        # -------------------------
        # RRF multi-model retrieval
        # -------------------------
        num_candidates_to_retrive = K_RERANKING * 5
        print(f"Retrieving {num_candidates_to_retrive} docs after RRF from all retrievers")
        candidates = retrieve_with_rrf(query, num_candidates_to_retrive)

        # -------------------------
        # Optional re-ranking
        # -------------------------
        if USE_RERANKING:
            final_results = rerank(
                query,
                candidates,
                top_k=K_RERANKING
            )["results"]
        else:
            final_results = candidates[:K_RERANKING]

        # -------------------------
        # Evaluation
        # -------------------------
        result = evaluate_query(final_results, item)
        print(result)
        rows.append(result)

    # -------------------------
    # Aggregate metrics
    # -------------------------
    metrics = {
        f"Recall@{K_RERANKING}": np.mean(
            [r[f"section_hit@{K_RERANKING}"] for r in rows]
        ),
        f"RemedyHit@{K_RERANKING}": np.mean(
            [r[f"remedy_hit@{K_RERANKING}"] for r in rows]
        ),
        "MRR": calc_mrr(rows),
        f"NDCG@{K_RERANKING}": np.mean(
            [r[f"ndcg@{K_RERANKING}"] for r in rows]
        ),
    }

    print("\nFINAL RRF METRICS")
    print(metrics)
    return metrics



# ------------------------------------------------------------
# Full benchmark: BM25 only
# ------------------------------------------------------------
def evaluate_query_set_bm25():
    test_set = load_test_queries(path=TEST_QUERIES_FINAL_DATA)
    rows = []

    bm25 = get_bm25()

    for item in test_set[:100]:
        question = item["query"]

        keyword_results = bm25.search(question, k=K_RERANKING)

        result = evaluate_query(keyword_results, item)
        print(result)
        rows.append(result)

    metrics = {
        f"Recall@{K_RERANKING}": np.mean([r[f"section_hit@{K_RERANKING}"] for r in rows]),
        f"RemedyHit@{K_RERANKING}": np.mean([r[f"remedy_hit@{K_RERANKING}"] for r in rows]),
        "MRR": calc_mrr(rows),
        f"NDCG@{K_RERANKING}": np.mean([r[f"ndcg@{K_RERANKING}"] for r in rows]),
    }

    print("\nBM25 FINAL METRICS")
    print(metrics)
    return metrics


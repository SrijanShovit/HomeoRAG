from src.config import K_RERANKING, K_RETRIEVAL
from src.reranker import rerank
from src.semantic_search import semantic_search


def query_phase():
    QUESTIONS = [
        "I have severe throat pain while swallowing, swollen glands and a dark red throat",
        "My urine burns and looks abnormal, and I have pain in my lower back",
        "I have sudden early-morning diarrhea that comes out in a hot, watery rush",
        "There is tremor, nervous weakness and back pain after illness"
    ]
    
    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 120)
        print(f"QUESTION {i}: {q}")
        print("=" * 120)

        response = semantic_search(q, k=K_RETRIEVAL)

        print(f"\nSearch latency: {response['latency_ms']} ms")
        print("-" * 120)

        # for rank, r in enumerate(response["results"], 1):
        #     print(f"\nRank {rank}")
        #     print("Medicine :", r["medicine"])
        #     print("Synonyms :", r["synonyms"])
        #     print("Score    :", round(r["score"], 4))
        #     print("Text     :", r["text"][:500])
        #     print("-" * 120)


        reranked = rerank(q, response["results"], top_k=K_RERANKING)

        print(f"Rerank latency       : {reranked['latency_ms']} ms")
        print("-" * 120)

        for rank, r in enumerate(reranked["results"], 1):
            print(f"\nRank {rank}")
            print("Medicine :", r["medicine"])
            print("Synonyms :", r["synonyms"])
            print("VecScore :", round(r["score"], 4))
            print("ReScore  :", round(r["rerank_score"], 4))
            print("Text     :", r["text"][:500])
            print("-" * 120)


        return reranked
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_PATH, EMBEDDING_MODEL
import time


def get_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )


def semantic_search(query: str, k: int = 5):
    """
    Run pure semantic similarity search over all indexed chunks.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    db = get_db()

    start = time.perf_counter()

    results = db.similarity_search_with_score(query, k=k)

    latency = (time.perf_counter() - start) * 1000

    output = []

    for doc, score in results:
        output.append({
            "medicine": doc.metadata.get("medicine"),
            "synonyms": doc.metadata.get("synonyms"),
            "score": float(score),
            "text": doc.page_content
        })

    return {
        "latency_ms": round(latency, 2),
        "results": output
    }


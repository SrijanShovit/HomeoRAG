from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
        # MS Marco
        PUBMED_BERT_MS_MARCO_CHROMA_1024_256_PATH, 
        PUBMED_BERT_MS_MARCO_EMBEDDING_MODEL,
        
        # All MiniLM
        SENTENCE_TF_ALL_MINILM_EMBEDDING_MODEL,
        SENTENCE_TF_ALL_MINILM_CHROMA_800_120_PATH,
        SENTENCE_TF_ALL_MINILM_CHROMA_768_128_PATH,
        
        # BAAI BGE
        BAAI_BGE_SMALL_EN_EMBEDDING_MODEL,
        BAAI_BGE_SMALL_EN_768_128_PATH,

        #BioBERT
        BIOBERT_768_128_PATH,
        BIOBERT_EMBEDDING_MODEL,

        #SapBERT
        SAPBERT_EMBEDDING_MODEL,
        SAPBERT_768_128_PATH,

        #BioMed PubMedBERT
        BIOMED_NLP_PUBMEDBERT_768_128_PATH,
        BIOMED_NLP_PUBMEDBERT_EMBEDDING_MODEL,

        #InFloat e5
        E5_BASE_768_128_PATH,
        E5_BASE_EMBEDDING_MODEL,


        K_RETRIEVAL
    )
import time


_EMBEDDINGS = None
_DB = None

def get_db():
    global _EMBEDDINGS, _DB
    if _DB is None:
        print("Loading embedding model once...")
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=E5_BASE_EMBEDDING_MODEL
        )
        _DB = Chroma(
            persist_directory=str(E5_BASE_768_128_PATH),
            embedding_function=_EMBEDDINGS
        )
    return _DB


def semantic_search(query: str, k: int = K_RETRIEVAL):
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


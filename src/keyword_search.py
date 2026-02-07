from langchain_community.retrievers import BM25Retriever
from src.config import BM25_DOCS_PATH, BM25_USE_NLTK
import pickle
from nltk.tokenize import word_tokenize

_BM25 = None

def get_bm25():
    global _BM25
    if _BM25 is None:
        _BM25 = BM25VectorDB()
    return _BM25


class BM25VectorDB:
    def __init__(self):
        with open(BM25_DOCS_PATH, "rb") as f:
            self.documents = pickle.load(f)

        kwargs = {}

        # ---- Tokenizer (optional) ----
        if BM25_USE_NLTK and word_tokenize:
            kwargs["preprocess_func"] = word_tokenize

        # ---- Build retriever ----
        self.retriever = BM25Retriever.from_documents(self.documents, **kwargs)

        print(
            f"BM25 loaded | "
            f"tokenizer={'nltk' if BM25_USE_NLTK else 'default'}"
        )

    def search(self, query, k):
        docs = self.retriever.invoke(query, k=k)

        # print(f"  -> BM25 retriever used. Requested k={k}, returned {len(docs)} docs")

        results = []
        for rank, doc in enumerate(docs):
            results.append({
                "id": doc.metadata["chunk_id"],
                "medicine": doc.metadata.get("medicine"),
                "synonyms": doc.metadata.get("synonyms"),
                "text": doc.page_content,
                "score": float(k - rank)   # RRF-friendly
            })
        return results

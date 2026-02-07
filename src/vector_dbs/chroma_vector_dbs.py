from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaVectorDB:
    def __init__(self, model_name, chroma_path):
        self.model_name = model_name
        self.chroma_path = chroma_path
        self._embeddings = None
        self._db = None

    def load(self):
        if self._db is None:
            print(f"Loading {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            self._db = Chroma(
                persist_directory=str(self.chroma_path),
                embedding_function=self._embeddings
            )
        return self._db

    def search(self, query, k):
        db = self.load()
        results = db.similarity_search_with_score(query, k=k)

        out = []
        for doc, score in results:
            out.append({
                "id": doc.metadata["chunk_id"],     # REQUIRED for RRF
                "medicine": doc.metadata.get("medicine"),
                "synonyms": doc.metadata.get("synonyms"),
                "score": float(score),
                "text": doc.page_content,
            })

        return out

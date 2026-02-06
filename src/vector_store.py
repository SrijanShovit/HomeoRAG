from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import PUBMED_BERT_MS_MARCO_CHROMA_PATH, PUBMED_BERT_MS_MARCO_EMBEDDING_MODEL
import time


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=PUBMED_BERT_MS_MARCO_EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(PUBMED_BERT_MS_MARCO_CHROMA_PATH),
        embedding_function=embeddings
    )


def get_count(db):
    try:
        return db._collection.count()
    except Exception:
        return 0
    

def save_documents(documents: list[Document]):
    db = get_vectorstore()

    # Extract IDs from metadata
    ids = [doc.metadata["chunk_id"] for doc in documents]

    # Check which IDs already exist
    existing_docs = db.get(ids=ids, include=[])
    existing_ids = set(existing_docs["ids"]) if existing_docs["ids"] else set()

    # Filter out already stored documents
    new_docs = [doc for doc in documents if doc.metadata["chunk_id"] not in existing_ids]
    new_ids = [doc.metadata["chunk_id"] for doc in new_docs]

    if not new_docs:
        print("All documents already exist. Skipping insert.")
        return

    # Insert only new docs
    before = get_count(db)
    print(f"Chroma before insert: {before}")

    start_time = time.perf_counter()
    db.add_documents(new_docs, ids=new_ids)
    persist_time = (time.perf_counter() - start_time)

    after = get_count(db)
    print(f"Chroma after insert: {after}")
    print(f"Inserted: {after - before}")
    print(f"Total ingest time: {persist_time:.2f} s")




from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import time
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
    )



def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=E5_BASE_EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(E5_BASE_768_128_PATH),
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




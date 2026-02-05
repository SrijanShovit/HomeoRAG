from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import CHROMA_PATH, EMBEDDING_MODEL


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )


def get_count(db):
    try:
        return db._collection.count()
    except Exception:
        return 0
    

def save_documents(documents:list[Document], ids:list[str]):
    db = get_vectorstore()

    before = get_count(db)
    print(f"\nChroma before insert: {before}")

    db.add_documents(documents, ids=ids)

    after = get_count(db)
    print(f"Chroma after insert: {after}")
    print(f"Inserted: {after - before}")



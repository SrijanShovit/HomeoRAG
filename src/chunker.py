from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
import hashlib



def make_chunk_id(medicine, synonyms, text):
    raw = f"{medicine}|{synonyms}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chunk_medicine(medicine: dict) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "––", ".", ";"]
    )

    chunks = splitter.split_text(medicine["content"])

    documents = []

    for chunk in chunks:
        chunk_id = make_chunk_id(
            medicine["name"],
            ", ".join(medicine["synonyms"]),
            chunk
        )

        chunk_document = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": chunk_id,
                    "medicine": medicine["name"],
                    "synonyms": ", ".join(medicine["synonyms"])
                }
            )
        
        documents.append(chunk_document)

    return documents

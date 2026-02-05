from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
import hashlib



def make_chunk_id(medicine, synonyms, text):
    raw = f"{medicine}|{synonyms}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chunk_medicine(medicine: dict) -> tuple[List[Document],List[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "––", ".", ";"]
    )

    chunks = splitter.split_text(medicine["content"])

    documents = []
    ids = []

    for chunk in chunks:
        chunk_id = make_chunk_id(
            medicine["name"],
            ", ".join(medicine["synonyms"]),
            chunk
        )
        ids.append(chunk_id)

        chunk_document = Document(
                page_content=chunk,
                metadata={
                    "medicine": medicine["name"],
                    "synonyms": ", ".join(medicine["synonyms"])
                }
            )
        
        documents.append(chunk_document)

    return documents,ids

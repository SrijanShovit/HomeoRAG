from src.config import DATA_PATH
from src.parser import split_medicines, extract_name_and_synonyms
from src.chunker import chunk_medicine
from src.vector_store import save_documents
from langchain_core.documents import Document
import random


def load_text():
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load data file: {e}")
    

def consolidate_all_medicine_chunks(medicine_blocks)->list[Document]:
    all_chunks = []

    for idx, block in enumerate(medicine_blocks):
        try:
            medicine = extract_name_and_synonyms(block)
            chunks = chunk_medicine(medicine)
            all_chunks.extend(chunks)

            print(f"✔ {medicine['name']} → {len(chunks)} chunks")

        except Exception as e:
            print(f"❌ Skipping block {idx}: {e}")

    return all_chunks

def show_sample_chunks(all_documents:list[Document],sample_size:int=5):
    if not all_documents:
        print("No documents available.")
        return

    total = len(all_documents)
    k = min(sample_size, total)

    indices = random.sample(range(total), k)

    print(f"\n--- Showing {k} random chunks out of {total} ---\n")

    for idx in indices:
        doc = all_documents[idx]
        print(f"Index: {idx}")
        print(f"Vector ID: {doc.metadata['chunk_id']}")
        print(f"Medicine: {doc.metadata['medicine']}")
        print(f"Synonyms: {doc.metadata['synonyms']}")
        print(f"Text: {doc.page_content[:300]}...")
        print("-" * 90)


def index_phase():
    print("Loading corpus...")
    raw_text = load_text()

    print("Splitting medicines...")
    medicine_blocks = split_medicines(raw_text)

    all_documents = consolidate_all_medicine_chunks(medicine_blocks)

    print(f"\nStoring {len(all_documents)} chunks in Chroma...")

    print("\n--- SAMPLE DOCUMENTS (before storing) ---\n")
    show_sample_chunks(all_documents)

    save_documents(all_documents)


    print("Done Indexing")
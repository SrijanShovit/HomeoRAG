import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.config import DATA_PATH
from src.parser import split_medicines, extract_name_and_synonyms
from src.chunker import chunk_medicine
from src.search import semantic_search
from src.vector_store import save_documents
from langchain_core.documents import Document
import random

def load_text():
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load data file: {e}")
    
def consolidate_all_medicine_chunks(medicine_blocks):
    all_documents = []
    all_ids = []

    for idx, block in enumerate(medicine_blocks):
        try:
            medicine = extract_name_and_synonyms(block)
            docs, ids = chunk_medicine(medicine)
            all_documents.extend(docs)
            all_ids.extend(ids)

            print(f"✔ {medicine['name']} → {len(docs)} chunks")

        except Exception as e:
            print(f"❌ Skipping block {idx}: {e}")

    return all_documents,all_ids

def show_sample_chunks(all_documents:list[Document],all_ids:list[str],sample_size:int=5):
    if not all_documents:
        print("No documents available.")
        return

    if len(all_documents) != len(all_ids):
        raise ValueError("Documents and IDs length mismatch")

    total = len(all_documents)
    k = min(sample_size, total)

    indices = random.sample(range(total), k)

    print(f"\n--- Showing {k} random chunks out of {total} ---\n")

    for idx in indices:
        doc = all_documents[idx]
        chunk_id = all_ids[idx]

        print(f"Index: {idx}")
        print(f"Vector ID: {chunk_id}")
        print(f"Medicine: {doc.metadata['medicine']}")
        print(f"Synonyms: {doc.metadata['synonyms']}")
        print(f"Text: {doc.page_content[:300]}...")
        print("-" * 90)


def index_phase():
    print("Loading corpus...")
    raw_text = load_text()

    print("Splitting medicines...")
    medicine_blocks = split_medicines(raw_text)

    all_documents,all_ids = consolidate_all_medicine_chunks(medicine_blocks)

    print(f"\nStoring {len(all_documents)} chunks in Chroma...")

    print("\n--- SAMPLE DOCUMENTS (before storing) ---\n")
    show_sample_chunks(all_documents,all_ids)

    
    save_documents(all_documents, all_ids)


    print("Done Indexing")





def query_phase():
    QUESTIONS = [
        "I have severe throat pain while swallowing, swollen glands and a dark red throat",
        "My urine burns and looks abnormal, and I have pain in my lower back",
        "I have sudden early-morning diarrhea that comes out in a hot, watery rush",
        "There is tremor, nervous weakness and back pain after illness"
    ]
    
    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 120)
        print(f"QUESTION {i}: {q}")
        print("=" * 120)

        response = semantic_search(q, k=5)

        print(f"\nSearch latency: {response['latency_ms']} ms")
        print("-" * 120)

        for rank, r in enumerate(response["results"], 1):
            print(f"\nRank {rank}")
            print("Medicine :", r["medicine"])
            print("Synonyms :", r["synonyms"])
            print("Score    :", round(r["score"], 4))
            print("Text     :", r["text"][:500])
            print("-" * 120)


if __name__ == "__main__":
    # index_phase()
    query_phase()

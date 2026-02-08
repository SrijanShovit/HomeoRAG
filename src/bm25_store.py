
import pickle
from src.config import BM25_DOCS_PATH
from src.parser import split_medicines
from src.vector_indexer import consolidate_all_medicine_chunks, load_text

def build_bm25_docs():
    print("Loading corpus...")
    raw_text = load_text()

    print("Splitting medicines...")
    medicine_blocks = split_medicines(raw_text)
    docs = consolidate_all_medicine_chunks(medicine_blocks)

    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"Saved {len(docs)} docs for BM25")
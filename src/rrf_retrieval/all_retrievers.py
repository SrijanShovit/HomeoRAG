from src.keyword_search import get_bm25
from src.vector_dbs.chroma_vector_dbs import ChromaVectorDB
from src.config import (
    E5_BASE_EMBEDDING_MODEL,E5_BASE_768_128_PATH,

    BAAI_BGE_SMALL_EN_EMBEDDING_MODEL,BAAI_BGE_SMALL_EN_768_128_PATH,

    SENTENCE_TF_ALL_MINILM_EMBEDDING_MODEL, SENTENCE_TF_ALL_MINILM_CHROMA_768_128_PATH, 

    PUBMED_BERT_MS_MARCO_EMBEDDING_MODEL, PUBMED_BERT_MS_MARCO_CHROMA_768_128_PATH
)

pubmed_chroma = ChromaVectorDB(PUBMED_BERT_MS_MARCO_EMBEDDING_MODEL,
                  PUBMED_BERT_MS_MARCO_CHROMA_768_128_PATH)

minilm_chroma = ChromaVectorDB(SENTENCE_TF_ALL_MINILM_EMBEDDING_MODEL,
                  SENTENCE_TF_ALL_MINILM_CHROMA_768_128_PATH)

bge_chroma = ChromaVectorDB(BAAI_BGE_SMALL_EN_EMBEDDING_MODEL,
               BAAI_BGE_SMALL_EN_768_128_PATH)

e5_chroma = ChromaVectorDB(E5_BASE_EMBEDDING_MODEL,
              E5_BASE_768_128_PATH)

bm25_word_tokenize = get_bm25()

RETRIEVER_POOL = {
    "pubmed_chroma": pubmed_chroma,
    "minilm_chroma": minilm_chroma,
    "bge_chroma": bge_chroma,
    "e5_chroma": e5_chroma,
    "bm25_word_tokenize": bm25_word_tokenize
}



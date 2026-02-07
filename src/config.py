
DATA_PATH = "data/boericke_materia_medica.txt"

# ----------------------------------------------------
#           Sentence Transformers All MiniLM L6 v2
# ----------------------------------------------------
SENTENCE_TF_ALL_MINILM_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


SENTENCE_TF_ALL_MINILM_CHROMA_768_128_PATH = "sentence_tf_all_minilm_chroma_768_128"
SENTENCE_TF_ALL_MINILM_CHROMA_800_120_PATH = "sentence_tf_all_minilm_chroma_800_120"


# ----------------------------------------------------
#           Pritamdeka S-PubMedBert-MS-MARCO
# ----------------------------------------------------
PUBMED_BERT_MS_MARCO_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


PUBMED_BERT_MS_MARCO_CHROMA_800_120_PATH = "pubmed_bert_ms_marco_chroma_800_120"
PUBMED_BERT_MS_MARCO_CHROMA_768_128_PATH = "pubmed_bert_ms_marco_chroma_768_128"
PUBMED_BERT_MS_MARCO_CHROMA_256_32_PATH = "pubmed_bert_ms_marco_chroma_256_32"
PUBMED_BERT_MS_MARCO_CHROMA_512_128_PATH = "pubmed_bert_ms_marco_chroma_512_128"
PUBMED_BERT_MS_MARCO_CHROMA_512_64_PATH = "pubmed_bert_ms_marco_chroma_512_64"
PUBMED_BERT_MS_MARCO_CHROMA_1024_128_PATH = "pubmed_bert_ms_marco_chroma_1024_128"
PUBMED_BERT_MS_MARCO_CHROMA_1024_256_PATH = "pubmed_bert_ms_marco_chroma_1024_256"


# ----------------------------------------------------
#           BAAI BGE Small en v1.5
# ----------------------------------------------------
BAAI_BGE_SMALL_EN_768_128_PATH = "baai_bge_small_en_768_128"

BAAI_BGE_SMALL_EN_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ----------------------------------------------------
#           Pritamdeka BiBERT
# ----------------------------------------------------
BIOBERT_768_128_PATH = "biobert_768_128"

BIOBERT_EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# ----------------------------------------------------
#           Cambridge LTL SapBERT
# ----------------------------------------------------
SAPBERT_768_128_PATH = "sapbert_768_128"

SAPBERT_EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# ----------------------------------------------------
#           Microsoft BioMed NLP PubMedBERT
# ----------------------------------------------------
BIOMED_NLP_PUBMEDBERT_768_128_PATH = "biomed_pubmedbert_768_128"

BIOMED_NLP_PUBMEDBERT_EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# ----------------------------------------------------
#           Infloat e5 base v2
# ----------------------------------------------------
E5_BASE_768_128_PATH = "e5_base_768_128"

E5_BASE_EMBEDDING_MODEL = "intfloat/e5-base-v2"

# ----------------------------------------------------
#           Chunking Settings
# ----------------------------------------------------
CHUNK_SIZE = 768
CHUNK_OVERLAP = 128

# ----------------------------------------------------
#           Retrieval & Ranking Settings
# ----------------------------------------------------

K_RETRIEVAL = 40
K_RERANKING = 5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

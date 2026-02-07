import os
import time
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.llm_response_generation import generate_llm_response_with_retrieved_context, normalize_llm_output
from src.rrf_retrieval.init_retrievers import init_all_retrievers
from src.rrf_retrieval.retrieval_pipeline import retrieve_with_rrf
from src.config import K_RERANKING

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="HomeoRAG – RRF Retrieval",
    layout="wide"
)

# -------------------------------
# Preload retrievers
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_retrievers():
    """Initialize all retrievers once at app startup."""
    return init_all_retrievers()

loaded_retrievers = load_retrievers()


# -------------------------------
# UI
# -------------------------------

st.title("🔬 HomeoRAG – RRF Retrieval Explorer")

st.markdown(
    """
This tool runs **Reciprocal Rank Fusion (RRF)** across multiple embedding models
and shows the **top retrieved Materia Medica chunks**.
"""
)

# -------------------------------
# Query input
# -------------------------------
query = st.text_input(
    "Enter patient query:",
    placeholder="e.g. burning stomach pain with sour vomiting"
)

# -------------------------------
# Search
# -------------------------------
if st.button("Search") and query.strip():
    with st.spinner("Running RRF retrieval..."):
        results = retrieve_with_rrf(
            query=query,
            loaded_retrievers=loaded_retrievers,
            reranking_k=K_RERANKING
        )

        llm_chain_response = generate_llm_response_with_retrieved_context(
            user_query=query,
            context_docs=results
        )

    # -------------------------------
    # LLM RESPONSE (PRIMARY)
    # -------------------------------
    st.subheader("🧠 Homeopathic Analysis")

    answer_text = normalize_llm_output(llm_chain_response)

    st.subheader("AI Answer")
    placeholder = st.empty()

    typed = ""
    for char in answer_text:
        typed += char
        placeholder.write(typed)
        time.sleep(0.01)

    # -------------------------------
    # RETRIEVAL EVIDENCE (ALWAYS SHOWN)
    # -------------------------------
    st.subheader("📚 Evidence from Materia Medica")

    if not results:
        st.info("No documents were retrieved for this query.")
    else:
        for i, doc in enumerate(results, start=1):
            with st.expander(f"#{i} — {doc.get('medicine', 'Unknown Remedy')}"):
                st.markdown(f"**Remedy:** {doc.get('medicine','-')}")
                st.markdown(f"**Synonyms:** {doc.get('synonyms','-')}")
                st.markdown("**Supporting Text:**")
                st.write(doc.get("text", ""))


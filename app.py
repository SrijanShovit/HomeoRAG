import os
import time
import streamlit as st

from src.observability.search_pipeline_tracing import run_search_pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.rrf_retrieval.init_retrievers import init_all_retrievers


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
        results , answer_text = run_search_pipeline(query,loaded_retrievers)

    # -------------------------------
    # LLM RESPONSE (PRIMARY)
    # -------------------------------
    st.subheader("🧠 Homeopathic Analysis")

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

   
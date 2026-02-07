import streamlit as st
from src.rrf_retrieval.retrieval_pipeline import retrieve_with_rrf
from src.config import K_RERANKING

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="HomeoRAG – RRF Retrieval",
    layout="wide"
)

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

top_k = st.slider("Number of chunks to show", 3, 15, K_RERANKING)

# -------------------------------
# Search
# -------------------------------
if st.button("Search") and query.strip():
    with st.spinner("Running RRF retrieval..."):
        results = retrieve_with_rrf(query, top_k)

    st.success(f"Retrieved {len(results)} chunks")

    # -------------------------------
    # Display results
    # -------------------------------
    for i, doc in enumerate(results, start=1):
        with st.expander(f"#{i} – {doc.get('medicine')}  |  RRF score = {round(doc.get('rrf_score',0),4)}"):
            st.markdown(f"**Medicine:** {doc.get('medicine')}")
            st.markdown(f"**Synonyms:** {doc.get('synonyms','-')}")
            st.markdown("**Text:**")
            st.write(doc.get("text"))

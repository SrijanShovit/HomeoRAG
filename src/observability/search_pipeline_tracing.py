from langsmith import traceable

from src.config import K_RERANKING
from src.observability.llm_response_generation import generate_llm_response_with_retrieved_context, normalize_llm_output
from src.rrf_retrieval.retrieval_pipeline import retrieve_with_rrf 

qwen_model_name = "llama-3.3-70b-versatile"

@traceable(name="streamlit_search_pipeline",tags=["ui", "homeorag",qwen_model_name])  # Traces entire pipeline
def run_search_pipeline(query: str,loaded_retrievers):
    """Full traceable pipeline: RRF retrieval + LLM response."""
    # ---------------------------- Retrieval ----------------------------
    results = retrieve_with_rrf(
        query=query,
        loaded_retrievers=loaded_retrievers,
        reranking_k=K_RERANKING
    )
    
    # ---------------------------- LLM API Call ----------------------------
    llm_chain_response = generate_llm_response_with_retrieved_context(
        user_query=query,
        model_name=qwen_model_name,
        context_docs=results
    )

    # ---------------------------- Result Content Extraction ----------------------------
    answer_text = normalize_llm_output(llm_chain_response)
    
    return results, answer_text
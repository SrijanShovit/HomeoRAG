from src.env_constants import GROQ_API_KEY
from langchain_groq import ChatGroq
from src.observability.langsmith_client import langsmith_client,prompt_version_id




def generate_llm_response_with_retrieved_context(user_query,model_name,context_docs):
    prompt = langsmith_client.pull_prompt(prompt_version_id)
    llm = ChatGroq(
            model=model_name,
            # reasoning_format="parsed",
            api_key=GROQ_API_KEY
        )

    chain = prompt | llm

    result = chain.invoke({
        "user_message": user_query,
        "context": context_docs
    })

    if result and result.content:
        return result.content
    
    fail_response_answer = "Sorry, there was some issue"
    
    return fail_response_answer


#### Response Utils

def normalize_llm_output(llm_output):
    """
    Converts whatever the chain returns into a displayable string.
    Handles str | list | dict | None safely.
    """
    if llm_output is None:
        return ""

    # If already a string
    if isinstance(llm_output, str):
        return llm_output.strip()

    # If LangChain message object
    if hasattr(llm_output, "content"):
        return str(llm_output.content).strip()

    # If list (e.g. messages, tool outputs, streaming chunks)
    if isinstance(llm_output, list):
        parts = []
        for item in llm_output:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            elif hasattr(item, "content"):
                parts.append(str(item.content))
        return "\n".join(parts).strip()

    # If dict
    if isinstance(llm_output, dict):
        return str(llm_output.get("content", "")).strip()

    return str(llm_output).strip()

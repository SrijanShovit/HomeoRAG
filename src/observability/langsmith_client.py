from langsmith import Client
from src.env_constants import LANGCHAIN_API_KEY

langsmith_client = Client(
        api_key=LANGCHAIN_API_KEY
    ) 

prompt_version_id = "homeopathy-rag-prompt"

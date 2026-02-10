from dotenv import load_dotenv
import os
from pydantic import SecretStr
load_dotenv()

GROQ_API_KEY = SecretStr(os.getenv("GROQ_API_KEY") or "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") or ""
LANGCHAIN_PROJECT = SecretStr(os.getenv("LANGCHAIN_PROJECT") or "")
LANGCHAIN_TRACING_V2 = SecretStr(os.getenv("LANGCHAIN_TRACING_V2") or "")
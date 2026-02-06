from dotenv import load_dotenv
import os
from pydantic import SecretStr
load_dotenv()

GROQ_API_KEY = SecretStr(os.getenv("GROQ_API_KEY") or "")
from src.observability.prompts import prompt_v2
from src.observability.langsmith_client import langsmith_client, prompt_version_id


def push_prompt_to_hub():
    
    url = langsmith_client.push_prompt(
        prompt_version_id,
        object=prompt_v2
    )
    print(url)

if __name__ == "__main__":
    push_prompt_to_hub()
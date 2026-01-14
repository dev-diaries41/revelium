import json
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from benchmarks.constants import BENCHMARK_CHROMADB_PATH, BENCHMARK_DIR
from revelium.constants.models import DEFAULT_SYSTEM_PROMPT, OPENAI_API_KEY, DEFAULT_OPENAI_MODEL
from revelium.prompts.prompts_manager import PromptsManager
from revelium.providers.llm.openai import OpenAIClient
from revelium.schemas.llm import LLMClientConfig
from revelium.prompts.prompts_manager import PromptsManager
from revelium.embeddings.helpers import get_embedding_store


BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "labelling_benchmarks.jsonl")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

async def run(revelium: PromptsManager, cluster_id: str, sample_size: int):
    result = revelium.label_prompts(cluster_id, sample_size)

    print(result)
    
    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        json.dump(result.model_dump(), f, indent=1)


async def main():
    prompt_embedding_store =  get_embedding_store(BENCHMARK_CHROMADB_PATH, PromptsManager.PROMPT_TYPE, 'all-minilm-l6-v2', 384) 
    cluster_embedding_store =  get_embedding_store(BENCHMARK_CHROMADB_PATH, PromptsManager.CLUSTER_TYPE, 'all-minilm-l6-v2', 384) 
    llm = OpenAIClient(OPENAI_API_KEY, LLMClientConfig(model_name=DEFAULT_OPENAI_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT))
    revelium = PromptsManager(llm_client=llm, prompt_embedding_store=prompt_embedding_store, cluster_embedding_store=cluster_embedding_store)
    await run(revelium, "fce4cfdc44b3ea3f", 10)

if __name__ == "__main__":
    asyncio.run(main())
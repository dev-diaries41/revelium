import json
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from benchmarks.constants import BENCHMARK_CHROMADB_PATH, BENCHMARK_PROMPT_STORE_PATH, BENCHMARK_DIR
from revelium.constants import DEFAULT_SYSTEM_PROMPT
from revelium.core.engine import Revelium, ReveliumConfig
from revelium.providers.llm.openai import OpenAIClient
from revelium.schemas.llm import LLMClientConfig
from revelium.constants.llms import OPENAI_API_KEY

BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "labelling_benchmarks.jsonl")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

async def run(revelium: Revelium, cluster_id: str, sample_size: int):
    result = revelium.label_prompts(cluster_id, sample_size)

    print(result)
    
    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        json.dump(result.model_dump(), f, indent=1)


async def main():
    llm = OpenAIClient(OPENAI_API_KEY, LLMClientConfig(model_name=ReveliumConfig.DEFAULT_OPENAI_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT))
    revelium = Revelium(config=ReveliumConfig(benchmarking=True, chromadb_path=BENCHMARK_CHROMADB_PATH, prompt_store_path=BENCHMARK_PROMPT_STORE_PATH), llm_client=llm)
    await run(revelium, "fce4cfdc44b3ea3f", 10)

if __name__ == "__main__":
    asyncio.run(main())
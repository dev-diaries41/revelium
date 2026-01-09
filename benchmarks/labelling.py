import json
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from revelium.api.local import Revelium, ReveliumConfig

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "labelling_benchmarks.jsonl")
BENCHMARK_PROMPT_STORE_PATH = os.path.join(BENCHMARK_DIR, "prompts.db")
BENCHMARK_CHROMADB_PATH = os.path.join(BENCHMARK_DIR, "chroma.db")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

async def run(revelium: Revelium, cluster_id: str, sample_size: int):
    result = await revelium.label_prompts(cluster_id, sample_size)
    
    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        json.dump(result.model_dump(), f, indent=1)


async def main():
    revelium_client = Revelium(config=ReveliumConfig(benchmarking=True, chromadb_path=BENCHMARK_CHROMADB_PATH, prompt_store_path=BENCHMARK_PROMPT_STORE_PATH))
    await run(revelium_client, "fce4cfdc44b3ea3f", 10)

if __name__ == "__main__":
    asyncio.run(main())
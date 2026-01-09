import json
import chromadb
import asyncio
import os
import argparse

from dotenv import load_dotenv

load_dotenv()

from revelium.api.local import Revelium
from benchmarks.helpers import get_collection_name, get_revelium_client

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
    client = chromadb.PersistentClient(path=BENCHMARK_CHROMADB_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
    collection_name = get_collection_name("minilm", 384)
    revelium_client = get_revelium_client(client, BENCHMARK_PROMPT_STORE_PATH, collection_name)
    await run(revelium_client, "fce4cfdc44b3ea3f", 10)

if __name__ == "__main__":
    asyncio.run(main())
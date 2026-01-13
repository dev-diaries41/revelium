import json
import asyncio
import os
import argparse

from dotenv import load_dotenv

load_dotenv()
from dataclasses import asdict
from revelium.prompts.types import Prompt
from revelium.data import get_dummy_data, get_placeholder_prompts
from revelium.core.engine import Revelium, ReveliumConfig
from revelium.prompts.indexer_listener import ProgressBarIndexerListener
from revelium.constants.llms import OPENAI_API_KEY
from benchmarks.constants import BENCHMARK_CHROMADB_PATH, BENCHMARK_PROMPT_STORE_PATH, BENCHMARK_DIR


BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "indexing_benchmarks.jsonl")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
async def run(revelium_client: Revelium, labelled_prompts: list[Prompt]):
    revelium_client.text_embedder.init()
    result =  await revelium_client.index_prompts(labelled_prompts)
    result_dict = {k: v for k, v in asdict(result).items() if k != "error"}
    print(f"{revelium_client.config.text_embedder}_result - time_elpased: {result.time_elapsed} | processed: {result.total_processed}")
    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(result_dict, indent=None) + "\n")


async def main(labelled_prompts: list[Prompt]):
    revelium = Revelium(config=ReveliumConfig(benchmarking=True, chromadb_path=BENCHMARK_CHROMADB_PATH, prompt_store_path=BENCHMARK_PROMPT_STORE_PATH))
    revelium.update_index_listener(ProgressBarIndexerListener())
    # openai_revelium_client = Revelium(config=ReveliumConfig(benchmarking=True, chromadb_path=BENCHMARK_CHROMADB_PATH, text_embedder="text-embedding-3-small", provider_api_key=OPENAI_API_KEY))
    await run(revelium, labelled_prompts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="number of items to generate", default=100)
    parser.add_argument("--stress", action="store_true", help="stress test")

    args = parser.parse_args()
    if args.n and args.stress:
        asyncio.run(main(get_dummy_data(args.n)))
    else:
        asyncio.run(main(get_placeholder_prompts()))

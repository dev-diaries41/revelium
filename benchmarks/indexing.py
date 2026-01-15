import json
import asyncio
import os
import argparse

from dotenv import load_dotenv

load_dotenv()
from dataclasses import asdict
from revelium.prompts.types import Prompt
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.indexer_listener import ProgressBarIndexerListener, PromptIndexListenerWithProgressBar
from revelium.data import get_dummy_data, get_placeholder_prompts
from revelium.prompts.prompts_manager import PromptsManager
from revelium.models.manage import ModelManager
from revelium.embeddings.helpers import get_embedding_store
from benchmarks.constants import BENCHMARK_CHROMADB_PATH, BENCHMARK_DIR
from revelium.constants.api import Routes
from revelium.constants.models import OPENAI_API_KEY, DEFAULT_SYSTEM_PROMPT, DEFAULT_OPENAI_MODEL
from revelium.constants import DEFAULT_CHROMADB_PATH
from revelium.prompts.cluster import cluster_prompts, get_cluster_plot
from revelium.models.manage import ModelManager
from revelium.embeddings.helpers import get_embedding_store
from revelium.prompts.indexer import PromptIndexer
from dotenv import load_dotenv

load_dotenv()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "indexing_benchmarks.jsonl")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
async def main(labelled_prompts: list[Prompt]):
    text_embedder = ModelManager().get_text_embedder('all-minilm-l6-v2')
    text_embedder.init()
    prompt_embedding_store =  get_embedding_store(BENCHMARK_CHROMADB_PATH, PromptsManager.PROMPT_TYPE, 'all-minilm-l6-v2', text_embedder.embedding_dim) 
    cluster_embedding_store =  get_embedding_store(BENCHMARK_CHROMADB_PATH, PromptsManager.CLUSTER_TYPE, 'all-minilm-l6-v2', text_embedder.embedding_dim) 
    prompts_manager = PromptsManager(prompt_embedding_store=prompt_embedding_store, cluster_embedding_store=cluster_embedding_store)

    indexer =  PromptIndexer(text_embedder, listener=PromptIndexListenerWithProgressBar(prompts_manager), embeddings_store=prompt_embedding_store, batch_size=100, max_concurrency=4)
    result =  await indexer.run(labelled_prompts)
    result_dict = {k: v for k, v in asdict(result).items() if k != "error"}
    print(f"result - time_elpased: {result.time_elapsed} | processed: {result.total_processed}")
    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(result_dict, indent=None) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="number of items to generate", default=100)
    parser.add_argument("--stress", action="store_true", help="stress test")

    args = parser.parse_args()
    if args.n and args.stress:
        asyncio.run(main(get_dummy_data(args.n)))
    else:
        asyncio.run(main(get_placeholder_prompts()))

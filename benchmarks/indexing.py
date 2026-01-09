import json
import chromadb
import asyncio
import os
import argparse

from dotenv import load_dotenv

load_dotenv()
from dataclasses import asdict
from smartscan.providers import  MiniLmTextEmbedder, TextEmbeddingProvider
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.indexer_listener import DefaultIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore, PromptStore
from revelium.prompts.types import Prompt
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.data import get_dummy_data, get_placeholder_prompts
from revelium.providers.embeddings.openai import OpenAITextEmbedder
from benchmarks.helpers import get_collection_name

from server.constants import MINILM_MODEL_PATH

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "indexing_benchmarks.jsonl")
BENCHMARK_PROMPT_STORE_PATH = os.path.join(BENCHMARK_DIR, "prompts.db")
BENCHMARK_CHROMADB_PATH = os.path.join(BENCHMARK_DIR, "chroma.db")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking


async def run(labelled_prompts: list[Prompt], embedders: dict[str, TextEmbeddingProvider]):
    client = chromadb.PersistentClient(path=BENCHMARK_CHROMADB_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
    prompt_store = AsyncSQLitePromptStore(BENCHMARK_PROMPT_STORE_PATH)
   
    results = {}
    for model, embedder in embedders.items():
        embedder.init()
        collection_name = get_collection_name(model, embedder.embedding_dim)
        collection = client.get_or_create_collection(name=collection_name)
        embedding_store = ChromaDBEmbeddingStore(collection)
        indexer = PromptIndexer(embedder, embedder._max_len, listener=DefaultIndexerListener(), prompt_store=prompt_store, embeddings_store=embedding_store, batch_size=100, max_concurrency=4)
        result =  await indexer.run(labelled_prompts)
        results[model] = {k: v for k, v in asdict(result).items() if k != "error"}
        print(f"{model}_result - time_elpased: {result.time_elapsed} | processed: {result.total_processed}")

    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(results, indent=None) + "\n")



async def main(labelled_prompts: list[Prompt]):
    openai_embedder = OpenAITextEmbedder(OPENAI_API_KEY)
    minilm_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH, 512)
    await run(labelled_prompts, {"minilm": minilm_embedder})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="number of items to generate", default=100)
    parser.add_argument("--stress", action="store_true", help="stress test")

    args = parser.parse_args()
    if args.n and args.stress:
        asyncio.run(main(get_dummy_data(args.n)))
    else:
        asyncio.run(main(get_placeholder_prompts()))

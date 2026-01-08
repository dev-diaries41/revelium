import json
import chromadb
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()
from dataclasses import asdict
from smartscan.providers import  MiniLmTextEmbedder, TextEmbeddingProvider
from revelium.prompts.prompt_indexer import PromptIndexer, DefaultPromptIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore, PromptStore
from revelium.prompts.types import Prompt
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.data import get_placeholder_prompts
from revelium.providers.embeddings.openai import OpenAITextEmbedder

from server.constants import MINILM_MODEL_PATH

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "indexing_benchmarks.jsonl")
BENCHMARK_PROMPT_STORE_PATH = os.path.join(BENCHMARK_DIR, "chroma.db")
BENCHMARK_CHROMADB_PATH = os.path.join(BENCHMARK_DIR, "prompts.db")
BENCHMARK_COLLECTION_PATH = os.path.join(BENCHMARK_DIR, "prompts.db")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking

async def run(labelled_prompts: list[Prompt], embedders: dict[str, TextEmbeddingProvider]):
    client = chromadb.PersistentClient(path=BENCHMARK_CHROMADB_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
    prompt_store = AsyncSQLitePromptStore(BENCHMARK_PROMPT_STORE_PATH)
   
    results = {}
    for name, embedder in embedders.items():
        embedder.init()
        collection_name = f"{name}_prompts_{embedder.embedding_dim}"
        collection = client.get_or_create_collection(name=collection_name)
        embedding_store = ChromaDBEmbeddingStore(collection)
        indexer = PromptIndexer(embedder, embedder._max_len, listener=DefaultPromptIndexerListener(), prompt_store=prompt_store, embeddings_store=embedding_store)
        result =  await indexer.run(labelled_prompts)
        results[name] = {k: v for k, v in asdict(result).items() if k != "error"}
        print(f"{name}_result - time_elpased: {result.time_elapsed} | processed: {result.total_processed}")

    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(results, indent=None) + "\n")



async def main(labelled_prompts: list[Prompt]):
    openai_embedder = OpenAITextEmbedder(OPENAI_API_KEY)
    minilm_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH, 512)
    await run(labelled_prompts, {"minilm": minilm_embedder})

asyncio.run(main(get_placeholder_prompts()))
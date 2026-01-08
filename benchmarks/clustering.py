import random
import json
import chromadb
import os

from dotenv import load_dotenv

load_dotenv()

from dataclasses import asdict
from smartscan import  ItemId
from smartscan.classify import IncrementalClusterer, calculate_cluster_accuracy
from revelium.data import get_placeholder_prompts
from revelium.prompts.types import Prompt
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.utils.decorators import with_time
from revelium.api.local import ReveliumLocalClient
from benchmarks.helpers import get_collection_name

from server.constants import MINILM_MODEL_PATH, DB_DIR

random.seed(32)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "clustering_benchmarks.jsonl")
BENCHMARK_PROMPT_STORE_PATH = os.path.join(BENCHMARK_DIR, "prompts.db")
BENCHMARK_CHROMADB_PATH = os.path.join(BENCHMARK_DIR, "chroma.db")

os.makedirs(BENCHMARK_DIR, exist_ok=True)


@with_time
def cluster(clusterer, query_result):
    return clusterer.cluster(query_result.ids, query_result.embeddings)


def run(labelled_prompts: list[Prompt], clusterer: IncrementalClusterer, clusters_info: dict[str, int]):
    client = chromadb.PersistentClient(path=BENCHMARK_CHROMADB_PATH, settings=chromadb.Settings(anonymized_telemetry=False))

    results = {}
      
    true_labels: dict[ItemId, str] = {}
    for p in labelled_prompts:
        true_labels[p.prompt_id] = p.prompt_id.split("_")[0]
    
    item_ids = [str(item_id) for item_id in true_labels.keys()]

    for model, dim in clusters_info.items():
        collection_name = get_collection_name(model, dim)
        collection = client.get_or_create_collection(name=collection_name)
        embedding_store = ChromaDBEmbeddingStore(collection)
        query_result = embedding_store.get(ids=item_ids, include=['embeddings', 'metadatas'])
        result, time = cluster(clusterer, query_result)
        acc_info = calculate_cluster_accuracy(true_labels, result.assignments)
        bench = {"accuracy": asdict(acc_info), "clustering_speed": time}
        results[model] = bench
        print(f"{model}: {bench}")

    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(results, indent=None) + "\n")

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
def main(labelled_prompts: list[Prompt]):
    clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.8, benchmarking=True)
    run(labelled_prompts, clusterer, {"minilm": 384})


main(get_placeholder_prompts())
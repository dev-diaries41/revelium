import random
import json
import chromadb
import os

from dotenv import load_dotenv

load_dotenv()

from dataclasses import asdict
from smartscan import  ItemId
from smartscan.classify.types import ClusterResult
from smartscan.classify import IncrementalClusterer, calculate_cluster_accuracy
from revelium.data import get_placeholder_prompts
from revelium.prompts.types import Prompt
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.utils.decorators import with_time
from benchmarks.helpers import get_collection_name

from server.constants import MINILM_MODEL_PATH, DB_DIR

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "clustering_benchmarks.jsonl")
BENCHMARK_ASSIGNMENTS_PATH = os.path.join(BENCHMARK_DIR, "assignments_clustering_benchmarks.jsonl")
BENCHMARK_PROMPT_STORE_PATH = os.path.join(BENCHMARK_DIR, "prompts.db")
BENCHMARK_CHROMADB_PATH = os.path.join(BENCHMARK_DIR, "chroma.db")

os.makedirs(BENCHMARK_DIR, exist_ok=True)


@with_time
def cluster(clusterer: IncrementalClusterer, query_result) -> tuple[ClusterResult, float]:
    return clusterer.cluster(query_result.ids, query_result.embeddings)

def run(labelled_prompts: list[Prompt], clusterer: IncrementalClusterer, clusters_info: dict[str, int]):
    true_labels: dict[ItemId, str] = {}
    for p in labelled_prompts:
        true_labels[p.prompt_id] = p.prompt_id.split("_")[0]
    item_ids = sorted(str(item_id) for item_id in true_labels.keys())

    results = {}

    for model, dim in clusters_info.items():
        clusterer.clear()
        # Option A: instantiate a fresh clusterer for full isolation
        # clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, benchmarking=True)
        ## NOTE: IncrementalClusterer uses random numbers internally. Running multiple models sequentially 
        # without reseeding causes non-deterministic clustering and lower accuracy. Reseed Python and 
        # before each clustering run to ensure reproducible results.

        random.seed(32)

        # Create a fresh client per-model to avoid internal cache collisions
        client = chromadb.PersistentClient(path=BENCHMARK_CHROMADB_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
        # Ensure collection name is unique per model/dim
        collection_name = get_collection_name(model, dim)
        collection = client.get_or_create_collection(name=collection_name)

        embedding_store = ChromaDBEmbeddingStore(collection)
        query_result = embedding_store.get(ids=item_ids, include=['embeddings', 'metadatas'])

        result, time = cluster(clusterer, query_result)
        acc_info = calculate_cluster_accuracy(true_labels, result.assignments)
        bench = {"accuracy": asdict(acc_info), "clustering_speed": time}
        results[collection_name] = bench
        print(f"{collection_name}: {bench}")

        with open(BENCHMARK_OUTPUT_PATH, "a") as f:
            f.write(json.dumps(results, indent=None) + "\n")

        # Save last result assignments
        with open(BENCHMARK_ASSIGNMENTS_PATH, "w") as f:
            json.dump(result.assignments, f, indent=1, sort_keys=True)

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
def main(labelled_prompts: list[Prompt]):
    clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, benchmarking=True)
    run(labelled_prompts, clusterer, {"minilm": 384, "openai": 1536})


main(get_placeholder_prompts())
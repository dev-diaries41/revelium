import random
import json
import chromadb
import os

from dotenv import load_dotenv

load_dotenv()

from dataclasses import asdict
from smartscan import  ItemId, ClusterResult
from smartscan.classify import IncrementalClusterer, calculate_cluster_accuracy
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.utils.decorators import with_time
from benchmarks.helpers import get_collection_name

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "clustering_benchmarks.jsonl")
BENCHMARK_ASSIGNMENTS_PATH = os.path.join(BENCHMARK_DIR, "assignments_clustering_benchmarks.jsonl")
BENCHMARK_PROMPT_STORE_PATH = os.path.join(BENCHMARK_DIR, "prompts.db")
BENCHMARK_CHROMADB_PATH = os.path.join(BENCHMARK_DIR, "chroma.db")

os.makedirs(BENCHMARK_DIR, exist_ok=True)


@with_time
def cluster(clusterer: IncrementalClusterer, ids, embeddings) -> tuple[ClusterResult, float]:
    return clusterer.cluster(ids, embeddings)


# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
def run(clusterer: IncrementalClusterer, clusters_info: dict[str, int]):
    results = {}

    for model, dim in clusters_info.items():
        clusterer.clear()
        ## NOTE: IncrementalClusterer uses random numbers internally. Running multiple models sequentially 
        # without reseeding causes non-deterministic clustering and lower accuracy. Reseed Python and 
        # before each clustering run to ensure reproducible results.

        random.seed(32)

        client = chromadb.PersistentClient(path=BENCHMARK_CHROMADB_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
        # Ensure collection name is unique per model/dim
        collection_name = get_collection_name(model, dim)
        collection = client.get_or_create_collection(name=collection_name)
        embedding_store = ChromaDBEmbeddingStore(collection)
        count = embedding_store.count()
        ids, embeddings = [], []

        limit = 500
        offset = 0

        while len(ids) < count:
            query_result = embedding_store.get(include=['embeddings'], offset=offset, limit=limit)
            ids.extend(query_result.ids)
            embeddings.extend(query_result.embeddings)
            offset += limit

        result, time = cluster(clusterer, ids, embeddings)
        # for c in result.clusters.values():
        #     print(c.metadata)

        true_labels: dict[ItemId, str] = {}
        for prompt_id in ids:
            label = prompt_id.split("_")[0]
            if not label: 
                print(f"[WARNING] {prompt_id} is not a valid labelled item.")
                continue
            true_labels[prompt_id] = label

        acc_info = calculate_cluster_accuracy(true_labels, result.assignments)
        bench = {"accuracy": asdict(acc_info), "clustering_speed": time}
        results[collection_name] = bench
        print(f"{collection_name}: {bench}")

        with open(BENCHMARK_OUTPUT_PATH, "a") as f:
            f.write(json.dumps(results, indent=None) + "\n")

        # Save last result assignments
        with open(BENCHMARK_ASSIGNMENTS_PATH, "w") as f:
            json.dump(result.assignments, f, indent=1, sort_keys=True)


def main():
    clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, benchmarking=True)
    run(clusterer, {"minilm": 384})


main()
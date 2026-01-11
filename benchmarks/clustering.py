import random
import json
import os
import asyncio

from dotenv import load_dotenv

load_dotenv()

from dataclasses import asdict
from smartscan import  ItemId, ClusterResult
from revelium.utils import with_time, get_new_filename
from revelium.core.engine import Revelium, ReveliumConfig
from benchmarks.constants import BENCHMARK_CHROMADB_PATH, BENCHMARK_PROMPT_STORE_PATH, BENCHMARK_DIR
from revelium.plot import plot_clusters
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "clustering_benchmarks.jsonl")
BENCHMARK_ASSIGNMENTS_PATH = os.path.join(BENCHMARK_DIR, "assignments_clustering_benchmarks.jsonl")
BENCHMARK_PLOTS_DIR = os.path.join(BENCHMARK_DIR, "plots")
BENCHMARK_CLUSTERS_PLOT =  "prompt_clusters"

os.makedirs(BENCHMARK_DIR, exist_ok=True)


@with_time
def cluster(revelium: Revelium) -> tuple[ClusterResult, float]:
    return revelium.cluster_prompts()


# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
async def run(revelium: Revelium, plot_output: str):
    results = {}
    revelium.clusterer.clear()
    ## NOTE: IncrementalClusterer uses random numbers internally. Running multiple models sequentially 
    # without reseeding causes non-deterministic clustering and lower accuracy. Reseed Python and 
    # before each clustering run to ensure reproducible results.

    random.seed(32)

    # Ensure collection name is unique per model/dim
    result, time = cluster(revelium)
    # for c in result.clusters.values():
    #     print(c.metadata)
    
    ## Assign clusters and update
    await revelium.update_prompts(result.assignments, result.merges)
    revelium.update_clusters(result.clusters, result.merges)

    # Plot to visualise prompt clusters
    ids, embeddings = revelium.get_all_prompt_embeddings()
    plot_clusters(ids, embeddings, result.assignments, output_path=plot_output)

    true_labels: dict[ItemId, str] = {}
    for prompt_id in result.assignments.keys():
        label = prompt_id.split("_")[0]
        if not label: 
            print(f"[WARNING] {prompt_id} is not a valid labelled item.")
            continue
        true_labels[prompt_id] = label

    acc_info = revelium.calculate_cluster_accuracy(true_labels, result.assignments)
    bench = {"accuracy": asdict(acc_info), "clustering_speed": time}
    results[revelium.config.text_embedder] = bench
    print(results)

    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(results, indent=None) + "\n")

    # Save last result assignments
    with open(BENCHMARK_ASSIGNMENTS_PATH, "w") as f:
        json.dump(result.assignments, f, indent=1, sort_keys=True)


async def main():
    revelium = Revelium(config=ReveliumConfig(benchmarking=True, chromadb_path=BENCHMARK_CHROMADB_PATH, prompt_store_path=BENCHMARK_PROMPT_STORE_PATH))
    plot_output = get_new_filename(BENCHMARK_PLOTS_DIR, BENCHMARK_CLUSTERS_PLOT, ".png")
    await run(revelium, plot_output)

asyncio.run(main())
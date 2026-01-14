import random
import json
import os

from dotenv import load_dotenv

load_dotenv()

from dataclasses import asdict
from smartscan import ClusterResult
from smartscan.classify import IncrementalClusterer
from revelium.utils import with_time, get_new_filename
from revelium.core.engine import Revelium, ReveliumConfig
from revelium.plot import plot_clusters
from benchmarks.constants import BENCHMARK_CHROMADB_PATH, BENCHMARK_PROMPT_STORE_PATH, BENCHMARK_DIR

BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "clustering_benchmarks.jsonl")
BENCHMARK_ASSIGNMENTS_PATH = os.path.join(BENCHMARK_DIR, "assignments_clustering_benchmarks.jsonl")
BENCHMARK_PLOTS_DIR = os.path.join(BENCHMARK_DIR, "plots")
BENCHMARK_CLUSTERS_PLOT =  "prompt_clusters"

os.makedirs(BENCHMARK_DIR, exist_ok=True)

@with_time
def cluster(clusterer: IncrementalClusterer, ids, embeddings) -> tuple[ClusterResult, float]:
    return clusterer.cluster(ids, embeddings)

# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
def run(revelium: Revelium, plot_output: str):
    results = {}
    ## NOTE: IncrementalClusterer uses random numbers internally. Running multiple models sequentially 
    # without reseeding causes non-deterministic clustering and lower accuracy. Reseed Python and 
    # before each clustering run to ensure reproducible results.

    random.seed(32)

    ids, embeddings, cluster_ids = revelium.get_all_prompt_embeddings()
    existing_clusters = revelium.get_all_clusters()
    existing_assignments = dict(zip(ids, cluster_ids))
    clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.9, merge_threshold=0.85, existing_assignments=existing_assignments, existing_clusters=existing_clusters)  
    result,time = cluster(clusterer, ids, embeddings)
    if ids and embeddings:
        # Plot to visualise prompt clusters
        plot_clusters(ids, embeddings, result.assignments, output_path=plot_output)
    acc_info = revelium.calculate_cluster_accuracy()
    bench = {"accuracy": asdict(acc_info), "clustering_speed": time}
    results[revelium.config.text_embedder] = bench
 
    if result.clusters:
        revelium.update_clusters(result.clusters, result.merges)

    if result.assignments:
        revelium.update_prompts(result.assignments, result.merges)
    print(results)

    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(results, indent=None) + "\n")

    # Save last result assignments
    with open(BENCHMARK_ASSIGNMENTS_PATH, "w") as f:
        json.dump(result.assignments, f, indent=1, sort_keys=True)


def main():
    revelium = Revelium(config=ReveliumConfig(benchmarking=True, chromadb_path=BENCHMARK_CHROMADB_PATH, prompt_store_path=BENCHMARK_PROMPT_STORE_PATH))
    plot_output = get_new_filename(BENCHMARK_PLOTS_DIR, BENCHMARK_CLUSTERS_PLOT, ".png")
    run(revelium, plot_output)

main()
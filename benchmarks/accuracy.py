import random
import json
import chromadb
import os

from dataclasses import asdict
from smartscan import  ItemId
from smartscan.classify import IncrementalClusterer, calculate_cluster_accuracy
from smartscan.providers import  MiniLmTextEmbedder

from revelium.data import get_placeholder_prompts
from revelium.prompts.types import Prompt
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore
from revelium.utils.decorators import with_time

from server.constants import MINILM_MODEL_PATH, DB_DIR

random.seed(32)

BENCHMARK_DIR = "output/benchmarks"
BENCHMARK_OUTPUT_PATH = os.path.join(BENCHMARK_DIR, "cluster_accuracy.jsonl")
os.makedirs(BENCHMARK_DIR, exist_ok=True)


# `prompt_id` must be prefixed with label e.g promptlabel_123
# this is only for benchmarking
@with_time
def main(labelled_prompts: list[Prompt]):
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH, 512)
    text_embedder.init()
    client = chromadb.PersistentClient(path=DB_DIR, settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=f"cluster_collection", metadata={"description": "Cluster Collection"})
    embedding_store = ChromaDBEmbeddingStore(collection)
    clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.8, benchmarking=True)
    
    true_labels: dict[ItemId, str] = {}
    for p in labelled_prompts:
        true_labels[p.prompt_id] = p.prompt_id.split("_")[0]
    
    item_ids = [str(item_id) for item_id in true_labels.keys()]
    query_result = embedding_store.get(ids=item_ids, include=['embeddings', 'metadatas'])
    result = clusterer.cluster(query_result.ids, query_result.embeddings)
    acc_info = calculate_cluster_accuracy(true_labels, result.assignments)

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    with open(BENCHMARK_OUTPUT_PATH, "a") as f:
        f.write(json.dumps(asdict(acc_info), indent=None) + "\n")

    print(acc_info)

main(get_placeholder_prompts())
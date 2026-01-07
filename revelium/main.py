import random
import json
import asyncio

from revelium.utils import with_time
from revelium.data import get_placeholder_prompts, get_label_counts, generate_test_clusters
from revelium.prompts.prompt_indexer import PromptIndexer, DefaultPromptIndexerListener
from revelium.prompts.store import PromptStore, AsyncSQLitePromptStore

from smartscan.cluster.analysis import calculate_cluster_accuracy, merge_clusters, count_predicted_labels, compare_clusters
from smartscan.cluster.incremental_clusterer import IncrementalClusterer
from smartscan.providers import  MiniLmTextEmbedder

from server.constants import MINILM_MODEL_PATH

## DEV ONLY


random.seed(32)

async def main():
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH, 512)
    text_embedder.init()

    prompt_store = AsyncSQLitePromptStore()
    count = await prompt_store.count()
    if count == 0:
        prompts = get_placeholder_prompts() # dev only placeholder
        await prompt_store.add(prompts)
    else:
        prompts = await prompt_store.get(limit=500, order_by="created_at")

    # Indexer
    indexer = PromptIndexer(text_embedder, 512, listener=DefaultPromptIndexerListener(), prompt_store=prompt_store)
    
    result = await indexer.run(prompts)
    print(f"time_elpased: {result.time_elapsed} | processed: {result.total_processed}")

    # Clustering
    cluster_manager = IncrementalClusterer(default_threshold=0.55, sim_factor=0.8)
    prototypes, assignments = cluster_manager.cluster(indexer.item_embeddings)
    # cluster_merges, assignments = generate_test_clusters(num_items=100_000, num_clusters=5_000)
    cluster_merges = compare_clusters(prototypes)    
        
    with open("output/assignments_long.json", "w") as f:
        json.dump(dict(sorted(assignments.items())), f, indent=1)

    assignments = merge_clusters(cluster_merges, assignments)

    with open("output/merged.json", "w") as f:
        json.dump(dict(sorted(assignments.items())), f, indent=1)

    # Metrics
    predict_count = count_predicted_labels(assignments, labels = ["physics", "quantum", "forex", "btc"])
    acc_info = calculate_cluster_accuracy(get_label_counts(), predict_count)
    print(acc_info.per_label, acc_info.mean_accuracy)



asyncio.run(main())
import random
import json
import asyncio
import chromadb

from smartscan.classify import calculate_cluster_accuracy, IncrementalClusterer
from smartscan.providers import  MiniLmTextEmbedder

from revelium.utils import with_time
from revelium.data import get_placeholder_prompts, get_label_counts, generate_test_clusters
from revelium.prompts.prompt_indexer import PromptIndexer, DefaultPromptIndexerListener
from revelium.prompts.store import AsyncSQLitePromptStore
from revelium.api.local import ReveliumLocalClient
from revelium.embeddings.chroma_store import ChromaDBEmbeddingStore

from server.constants import MINILM_MODEL_PATH, DB_DIR

## DEV ONLY


random.seed(32)


async def main():
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH, 512)
    text_embedder.init()
    prompt_store = AsyncSQLitePromptStore("db/prompts.db")
    client = chromadb.PersistentClient(path=DB_DIR, settings=chromadb.Settings(anonymized_telemetry=False))

    collection = client.get_or_create_collection(
            name=f"cluster_collection",
            metadata={"description": "Cluster Collection"}
            )
    embedding_store = ChromaDBEmbeddingStore(collection)
    indexer = PromptIndexer(text_embedder, 512, listener=DefaultPromptIndexerListener(), prompt_store=prompt_store, embeddings_store=embedding_store)
    clusterer = IncrementalClusterer(default_threshold=0.55, sim_factor=0.8)
    revelium = ReveliumLocalClient(text_embedder, prompt_store, embedding_store, indexer , clusterer)

    count = await revelium.prompt_store.count()
    if count == 0:
        prompts = get_placeholder_prompts() # dev only placeholder
        await revelium.prompt_store.add(prompts)
    else:
        prompts = await revelium.prompt_store.get(limit=500, order_by="created_at")

    print(count)

    # Indexer
    result = await revelium.indexer.run(prompts)
    print(f"time_elpased: {result.time_elapsed} | processed: {result.total_processed}")

    # Clustering
    # TODO 
    query_result = revelium.embedding_store.get(include=['embeddings'])
    result = revelium.cluster(query_result.ids, query_result.embeddings)
    # cluster_merges, assignments = generate_test_clusters(num_items=100_000, num_clusters=5_000)
        
    with open("output/assignments_long.json", "w") as f:
        json.dump(dict(sorted(result.assignments.items())), f, indent=1)

    # if result.merges
    # with open("output/merged.json", "w") as f:
    #     json.dump(dict(sorted(assignments.items())), f, indent=1)

    # Metrics
    labelled_cluster_counts= get_label_counts()
    cluster_ids = labelled_cluster_counts.keys()
    predict_count = {}
    for cluster_id in cluster_ids:
        predict_count[cluster_id] = await revelium.prompt_store.count(cluster_id=cluster_id)
    acc_info = calculate_cluster_accuracy(labelled_cluster_counts, predict_count)
    print(acc_info.per_label, acc_info.mean_accuracy)



asyncio.run(main())
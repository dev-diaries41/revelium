import random
import json
import asyncio
import numpy as np

from revelium.utils import with_time
from revelium.data import get_prompts
from smartscan import Prototype
from revelium.indexer import PromptIndexer, DefaultPromptIndexerListener
from smartscan.cluster import IncrementalClusterer
from smartscan.providers import  MiniLmTextEmbedder
from server.constants import MINILM_MODEL_PATH


random.seed(32)


def compare_clusters(prototypes: dict[str, Prototype], merge_threshold: float = 0.9, verbose:bool = False) ->  dict[str, list[str]]:
    cluster_merges: dict[str, list[str]] = {}
    p_ids, p_embeds =  zip(*((p.prototype_id, p.embedding) for p in prototypes.values()))

    for idx, emb in enumerate(p_embeds):
        # compute dot product similarity with all embeddings
        sims = np.dot(p_embeds, emb)
        # exclude self-similarity
        sims_no_self = np.delete(sims, idx)
        if verbose: print(f"Embedding {p_ids[idx]} similarities with others: {sims_no_self}")
        
        merge_cluster_id_indices = [int(np.where(sims_no_self == sim)[0][0]) for sim in sims_no_self if sim > merge_threshold]  
        if(len(merge_cluster_id_indices)) > 0:
            cluster_merges[p_ids[idx]] = [p_ids[cluster_idx] for cluster_idx in merge_cluster_id_indices]
    
    return cluster_merges

def merge_clusters(cluster_merges: dict[str, list[str]], assignments: dict[str, str]):
    for item_id, assigned_cluster_id in assignments.items():
        for merge_cluster_id, cluster_ids_to_merge in cluster_merges.items():
            if assigned_cluster_id in cluster_ids_to_merge:
                assignments[item_id] = merge_cluster_id
    return assignments

async def main():
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
    text_embedder.init()

    # Indexer
    indexer = PromptIndexer(text_embedder, 512, listener=DefaultPromptIndexerListener())
    prompts = get_prompts() # dev only placeholder
    result = await indexer.run(prompts)
    print(f"time_elpased: {result.time_elapsed} | processed: {result.total_processed}")

    # Clustering
    cluster_manager = IncrementalClusterer(default_threshold=0.35, sim_factor=0.8)
    prototypes, assignments = cluster_manager.cluster(indexer.item_embeddings)
   
    # Analysis
    p_ids, p_embeds =  zip(*((p.prototype_id, p.embedding) for p in prototypes.values()))
    cluster_merges = compare_clusters(prototypes)    
        
    with open("output/assignments_long.json", "w") as f:
        json.dump(dict(sorted(assignments.items())), f, indent=1)

    assignments = merge_clusters(cluster_merges, assignments)

    with open("output/merged.json", "w") as f:
        json.dump(dict(sorted(assignments.items())), f, indent=1)




asyncio.run(main())
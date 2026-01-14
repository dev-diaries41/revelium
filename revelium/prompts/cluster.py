from smartscan.classify import IncrementalClusterer
from revelium.core.engine import Revelium

def cluster_prompts(revelium: Revelium):
    ids, embeddings, cluster_ids = revelium.get_all_prompt_embeddings()
    existing_clusters = revelium.get_all_clusters()
    existing_assignments = dict(zip(ids, cluster_ids))
    clusterer = IncrementalClusterer(
        default_threshold=0.55,
        sim_factor=0.9,
        merge_threshold=0.85,
        existing_assignments=existing_assignments,
        existing_clusters=existing_clusters,
    )
    return clusterer.cluster(ids, embeddings)

from smartscan.classify import IncrementalClusterer
from revelium.prompts.prompts_manager import PromptsManager

def cluster_prompts(prompts_manager: PromptsManager):
    ids, embeddings, cluster_ids = prompts_manager.get_all_prompt_embeddings()
    existing_clusters = prompts_manager.get_all_clusters()
    existing_assignments = dict(zip(ids, cluster_ids))
    clusterer = IncrementalClusterer(
        default_threshold=0.55,
        sim_factor=0.9,
        merge_threshold=0.85,
        existing_assignments=existing_assignments,
        existing_clusters=existing_clusters,
    )
    return clusterer.cluster(ids, embeddings)

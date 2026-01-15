import random
import string
from placeholder_data import physics_sentences, quantum_mechanics_sentences, btc_analysis, forex_analysis, long_physics_sentences, long_btc_analysis, long_forex_analysis
from revelium.prompts.types import Prompt
from revelium.providers.llm.llm_client import LLMClient


from pydantic import BaseModel
from typing import List

class PromptContent(BaseModel):
    prompts: List[str]
    
def generate_synthetic_prompt_data(llm: LLMClient, label: str, topic: str, n:int, prompt_size:int = 300, prompt: str | None = None):
    prompt = prompt or f"""Generate {n} realistic example prompts, each {prompt_size} characters long, that a user may have about {topic}"""
    result =  llm.generate_json(prompt, PromptContent)
    return strings_to_prompts(result.prompts, label)

## DEV ONLY placeholders for getting data to cluster
def strings_to_prompts(arr: list[str], label_prefix: str, offset: int = 0) -> list[Prompt]:
    return [Prompt(prompt_id=f"{label_prefix}_{idx + offset}", content=prompt_content) for idx, prompt_content in enumerate(arr)]

def get_placeholder_prompts(offset=0) -> list[Prompt]:
    all_data: list[Prompt] = []
    all_data.extend(strings_to_prompts(long_physics_sentences, "physics", offset))
    all_data.extend(strings_to_prompts(quantum_mechanics_sentences, "quantum", offset))
    all_data.extend(strings_to_prompts(long_btc_analysis, "btc", offset))
    all_data.extend(strings_to_prompts(long_forex_analysis, "forex", offset))
    return all_data

def get_dummy_data(n: int = 100, offset = 0) -> list[Prompt]:
    return [ Prompt(prompt_id = f"prompt_{i + offset}", content=random_string(550)).model_dump() for i in range(n)]

def random_string(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def generate_test_clusters(num_items: int = 100_000, num_clusters: int = 1_000, max_merges_per_cluster: int = 5,) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Generate random cluster_merges and assignments for benchmarking.

    Returns:
        cluster_merges: dict mapping target_cluster_id -> list of cluster_ids to merge
        assignments: dict mapping item_id -> cluster_id
    """
    def random_id(length: int = 8) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    # Generate cluster ids
    all_cluster_ids = [random_id() for _ in range(num_clusters)]

    # Generate cluster_merges
    cluster_merges = {}
    for target_id in all_cluster_ids[:num_clusters // 10]:  # 10% as merge targets
        merge_candidates = random.sample(all_cluster_ids, k=min(max_merges_per_cluster, num_clusters))
        cluster_merges[target_id] = merge_candidates

    # Generate assignments
    assignments = {}
    for i in range(num_items):
        item_id = f"item_{i}"
        cluster_id = random.choice(all_cluster_ids)
        assignments[item_id] = cluster_id

    return cluster_merges, assignments



from const import physics_sentences, quantum_mechanics_sentences, btc_analysis, forex_analysis, long_physics_sentences, long_btc_analysis, long_forex_analysis
## DEV ONLY placeholders for getting data to cluster
def arr_with_id(arr: list[str], id_prefix: str) -> list[tuple[str, str]]:
    return [(f"{id_prefix}_{idx}", item) for idx, item in enumerate(arr)]

def get_prompts() -> list[tuple[str, str]]:
    all_data: list[tuple[str, str]] = []
    all_data.extend(arr_with_id(long_physics_sentences, "physics"))
    all_data.extend(arr_with_id(quantum_mechanics_sentences, "quantum"))
    all_data.extend(arr_with_id(long_btc_analysis, "btc"))
    all_data.extend(arr_with_id(long_forex_analysis, "forex"))
    return all_data

import random
import json
import asyncio
import numpy as np

from smartscan import ItemEmbedding
from smartscan.utils import chunk_text
from smartscan.embeddings import generate_prototype_embedding
from smartscan.cluster import IncrementalClusterer
from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.providers import TextEmbeddingProvider, MiniLmTextEmbedder
from server.constants import MINILM_MODEL_PATH
from const import physics_sentences, quantum_mechanics_sentences, btc_analysis, forex_analysis, long_physics_sentences, long_btc_analysis, long_forex_analysis
from revelium.utils import with_time

random.seed(32)

class TextIndexerListener(ProcessorListener[tuple[str, str], ItemEmbedding]):
    def on_error(self, e, item):
        print(e)

class TextIndexer(BatchProcessor[tuple[str, str], ItemEmbedding]):
    def __init__(self, 
                text_encoder: TextEmbeddingProvider,
                max_tokenizer_length: int,
                max_chunks: int | None = None,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.max_chunks = max_chunks
        self.max_tokenizer_length = max_tokenizer_length
        self.item_embeddings = []

    # All chunks share the same item_id (url or file) so that chunks are group
    # In the on_batch_complete method, the listener can handle use it as metaddata and assign unique ids to each chunk if required
    def on_process(self, item):
        chunks = chunk_text(item[1], self.max_tokenizer_length)
        embeddings = self.text_encoder.embed_batch(chunks)
        text_prototype = generate_prototype_embedding(embeddings)
        return ItemEmbedding(item[0], text_prototype)
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        self.item_embeddings.extend( batch)
        

def arr_with_id(arr: list[str], id_prefix: str) -> list[tuple[str, str]]:
    return [(f"{id_prefix}_{idx}", item) for idx, item in enumerate(arr)]


async def main_clster():
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
    text_embedder.init()
    all_data: list[tuple[str, str]] = []
    all_data.extend(arr_with_id(long_physics_sentences, "physics"))
    all_data.extend(arr_with_id(quantum_mechanics_sentences, "quantum"))
    all_data.extend(arr_with_id(long_btc_analysis, "btc"))
    all_data.extend(arr_with_id(long_forex_analysis, "forex"))

    indexer = TextIndexer(text_embedder, 512, listener=TextIndexerListener())
    result = await indexer.run(all_data)
    print(f"time_elpased: {result.time_elapsed} | processed: {result.total_processed}")
    cluster_manager = IncrementalClusterer(default_threshold=0.35, sim_factor=0.8)
    prototypes, assignments = cluster_manager.cluster(indexer.item_embeddings)
    p_embeds = [p.embedding for p in prototypes.values()]
   
    # for i, emb in enumerate(p_embeds):
    #     # compute dot product similarity with all embeddings
    #     sims = np.dot(p_embeds, emb)
    #     # exclude self-similarity
    #     sims_no_self = np.delete(sims, i)
    #     print(f"Embedding {i} similarities with others: {sims_no_self}")
        
    with open("assignments_long.json", "w") as f:
        json.dump(dict(sorted(assignments.items())), f, indent=1)


asyncio.run(main_clster())
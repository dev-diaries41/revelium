import asyncio
import numpy as np

from smartscan.utils import chunk_text
from smartscan.providers import MiniLmTextEmbedder
from smartscan.embeddings import generate_prototype_embedding, calculate_cohesion_score
from server.constants import MINILM_MODEL_PATH
from revelium.semantic_filter import SemanticContentFilter, SemanticContentFilterListener
from const import yt_comments, physics_sentences, quantum_mechanics_sentences, entropy_sentences, btc_analysis, forex_analysis


async def main():
    batch_size = 10
    youtube_comments = [(idx, comment) for idx, comment in enumerate(yt_comments)]
    youtube_comments_batched = [youtube_comments[i:i+batch_size] for i in range(0, len(youtube_comments), batch_size)]
    
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
    text_embedder.init()

    criteria = "suggestions for new video topics"
    criteria_embed = text_embedder.embed(criteria)

    content_filter = SemanticContentFilter(text_embedder=text_embedder, threshold=0.29, criteria_embedding=criteria_embed, listener = SemanticContentFilterListener())
    result = await content_filter.run(youtube_comments_batched)
    comms = [youtube_comments[_id][1] for _id in content_filter.match_ids if youtube_comments[_id][0] in content_filter.match_ids]
    # print(comms)
    print(f" Time elapsed: {result.time_elapsed:.3f} s | Processed: {result.total_processed}")

async def test():
    text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
    text_embedder.init()

    batch_embeds = text_embedder.embed_batch(entropy_sentences[:15])
    prototype = generate_prototype_embedding(batch_embeds)
    c_score = calculate_cohesion_score(prototype, batch_embeds)

    single_embed = text_embedder.embed(entropy_sentences[19])
    sims = np.dot(prototype, single_embed)

    print(f" Similarity: {sims} | Cohesions: {c_score}")

    

async def prompt_test():
    # text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
    # text_embedder.init()

    # forex_batch_embeds = text_embedder.embed_batch(forex_analysis[:8])
    # forex_prototype = generate_prototype_embedding(forex_batch_embeds)

    # forex_single_embed = text_embedder.embed(forex_analysis[9])
    # btc_single_embed = text_embedder.embed(btc_analysis[9])

    # forex_sim = np.dot(forex_prototype, forex_single_embed)
    # print(f"Forex Similarity: {forex_sim}")
    # btc_sim = np.dot(forex_prototype, btc_single_embed)
    # print(f"Btc Similarity: {btc_sim}")
    chunks = chunk_text(physics_sentences[0], 128)
    print(len(physics_sentences[0]))

asyncio.run(prompt_test())
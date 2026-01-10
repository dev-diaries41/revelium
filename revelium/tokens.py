import tiktoken
from revelium.providers.types import TextEmbeddingModel
from smartscan.providers import MiniLmTextEmbedder
from revelium.constants import MINILM_MODEL_PATH


def count_tokens_embedding(text: str, model: TextEmbeddingModel) -> int:
    if model == "all-minilm-l6-v2":
        text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
        padded_tkns = text_embedder._tokenize(text)
        return len([token for token in padded_tkns if token != 0])
    else:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))


def embedding_token_cost(text: str, price_per_1m_tokens: float, model: TextEmbeddingModel) -> float:
    tokens = count_tokens_embedding(text, model)
    return tokens * (price_per_1m_tokens / 1_000_000)
    


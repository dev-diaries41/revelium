import tiktoken
from smartscan import ModelName
from smartscan.providers import MiniLmTextEmbedder
from server.constants import MINILM_MODEL_PATH


def count_tokens_embedding(text: str, model: str | ModelName = "text-embedding-3-large") -> int:
    if model == "all-minilm-l6-v2":
        text_embedder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
        padded_tkns = text_embedder._tokenize(text)
        return len([token for token in padded_tkns if token != 0])
    else:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))


def embedding_token_cost(text: str, price_per_1m_tokens: float, model: str | ModelName = "text-embedding-3-large") -> float:
    tokens = count_tokens_embedding(text, model)
    return tokens * (price_per_1m_tokens / 1_000_000)
    


from smartscan import ItemEmbedding
from smartscan.media import chunk_text
from smartscan.embeds import generate_prototype_embedding, EmbeddingStore
from smartscan.processor import BatchProcessor
from smartscan.providers import TextEmbeddingProvider
from revelium.prompts.store import PromptStore
from revelium.prompts.types import Prompt

class PromptIndexer(BatchProcessor[Prompt, ItemEmbedding]):
    def __init__(self, 
                text_encoder: TextEmbeddingProvider,
                max_tokenizer_length: int,
                prompt_store: PromptStore,
                embeddings_store: EmbeddingStore,
                max_chunks: int | None = None,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.max_chunks = max_chunks
        self.max_tokenizer_length = max_tokenizer_length
        self.prompt_store = prompt_store
        self.embeddings_store = embeddings_store

    # All chunks share the same item_id (url or file) so that chunks are group
    # In the on_batch_complete method, the listener can handle use it as metaddata and assign unique ids to each chunk if required
    def on_process(self, item):
        chunks = chunk_text(item.content, self.max_tokenizer_length)
        embeddings = self.text_encoder.embed_batch(chunks)
        text_prototype = generate_prototype_embedding(embeddings)
        return ItemEmbedding(item.prompt_id, text_prototype)
             
    async def on_batch_complete(self, batch):
        self.embeddings_store.add(batch)
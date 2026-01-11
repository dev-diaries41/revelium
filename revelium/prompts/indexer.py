from smartscan import ItemEmbedding
from smartscan.media import chunk_text
from smartscan.embeds import generate_prototype_embedding, EmbeddingStore
from smartscan.processor import BatchProcessor
from smartscan.providers import TextEmbeddingProvider
from revelium.prompts.types import Prompt

class PromptIndexer(BatchProcessor[Prompt, ItemEmbedding]):
    def __init__(self, 
                text_encoder: TextEmbeddingProvider,
                embeddings_store: EmbeddingStore,
                max_chunks: int | None = None,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.max_chunks = max_chunks
        self.max_tokenizer_length = text_encoder.max_tokenizer_length
        self.embeddings_store = embeddings_store

    # All chunks share the same item_id (url or file) so that chunks are group
    # In the on_batch_complete method, the listener can handle use it as metaddata and assign unique ids to each chunk if required
    def on_process(self, item):
        chunks = chunk_text(item.content, self.max_tokenizer_length)
        embeddings = self.text_encoder.embed_batch(chunks)
        text_prototype = generate_prototype_embedding(embeddings)
        return ItemEmbedding(item.prompt_id, text_prototype, data=item.content, metadata=item.metadata.model_dump())
             
    async def on_batch_complete(self, batch):
        if len(batch) == 0:
            return
        self.embeddings_store.add(batch)
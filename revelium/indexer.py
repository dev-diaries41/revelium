from smartscan import ItemEmbedding
from smartscan.utils import chunk_text
from smartscan.embeddings import generate_prototype_embedding
from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.providers import TextEmbeddingProvider

class PromptIndexer(BatchProcessor[tuple[str, str], ItemEmbedding]):
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
        

class DefaultPromptIndexerListener(ProcessorListener[tuple[str, str], ItemEmbedding]):
    def on_error(self, e, item):
        print(e)

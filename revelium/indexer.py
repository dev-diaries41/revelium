from smartscan import ItemEmbedding
from smartscan.utils import chunk_text
from smartscan.embeddings import generate_prototype_embedding
from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.providers import TextEmbeddingProvider
from revelium.prompts.store import PromptStore

class PromptIndexer(BatchProcessor[str, ItemEmbedding]):
    def __init__(self, 
                text_encoder: TextEmbeddingProvider,
                max_tokenizer_length: int,
                prompt_store: PromptStore,
                max_chunks: int | None = None,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.max_chunks = max_chunks
        self.max_tokenizer_length = max_tokenizer_length
        self.item_embeddings = []
        self.prompt_store = prompt_store

    # All chunks share the same item_id (url or file) so that chunks are group
    # In the on_batch_complete method, the listener can handle use it as metaddata and assign unique ids to each chunk if required
    def on_process(self, item):
        prompts = self.prompt_store.get_by_ids(ids=[item])
        if len(prompts) == 0:
            raise ValueError("Prompt not found")
        
        prompt = prompts[0]
        chunks = chunk_text(prompt.content, self.max_tokenizer_length)
        embeddings = self.text_encoder.embed_batch(chunks)
        text_prototype = generate_prototype_embedding(embeddings)
        return ItemEmbedding(prompt.prompt_id, text_prototype)
             
    # TODO: add to EmbeddingStore
    async def on_batch_complete(self, batch):
        self.item_embeddings.extend( batch)
        

class DefaultPromptIndexerListener(ProcessorListener[str, ItemEmbedding]):
    def on_error(self, e, item):
        print(e)

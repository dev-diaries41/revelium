from fastapi import WebSocket

from smartscan.processor import ProcessorListener
from smartscan.processor.types import Input, Output
from smartscan.types import ItemEmbedding
from smartscan.embeds import EmbeddingStore

from revelium.schemas.api import FailMessage, ErrorMessage, ProgressMessage, CompleteMessage, ActiveMessage



class BaseWebSocketListener(ProcessorListener[Input, Output]):
    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def on_active(self):
        await self.ws.send_json(ActiveMessage().model_dump())  


    async def on_progress(self, progress):
        await self.ws.send_json(ProgressMessage(progress=progress).model_dump())  
    
    
    async def on_fail(self, result):
        await self.ws.send_json(FailMessage(error=str(result.error)).model_dump())


    async def on_error(self, e, item):
        await self.ws.send_json(ErrorMessage(error=str(e), item=item).model_dump())


    async def on_complete(self, result):
        await self.ws.send_json(CompleteMessage(total_processed=result.total_processed, time_elapsed=result.time_elapsed).model_dump())



class PromptIndexerWebSocketListener(ProcessorListener[str, ItemEmbedding]):
    def __init__(self, cluster_embedding_store: EmbeddingStore, kwargs):
        super().__init__(**kwargs)
        self.cluster_embedding_store = cluster_embedding_store

    async def on_batch_complete(self, batch):
        if len(batch) <= 0:
            return
        self.cluster_embedding_store.add(batch)
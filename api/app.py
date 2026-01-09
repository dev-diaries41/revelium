import chromadb
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException,  WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from smartscan.constants import SupportedFileTypes
from smartscan.utils import  get_files_from_dirs
from smartscan.indexer import DocIndexer
from smartscan.providers import MiniLmTextEmbedder
from server.indexer import FileIndexerWebSocketListener, FailMessage
from revelium.constants import  DB_DIR, MINILM_MODEL_PATH

client = chromadb.PersistentClient(path=DB_DIR, settings=chromadb.Settings(anonymized_telemetry=False))

text_store = client.get_or_create_collection(
        name=f"prompts_collection",
        metadata={"description": "Prompt Collection"}
        )

text_encoder = MiniLmTextEmbedder(MINILM_MODEL_PATH)
text_encoder.init()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

class TextQueryRequest(BaseModel):
    query: str
    threshold: float = 0.9


async def _text_query(request: TextQueryRequest, store: chromadb.Collection):
    if request.query is None:
        raise HTTPException(status_code=400, detail="Missing query text")
  
    try:
        query_embedding = await run_in_threadpool(text_encoder.embed, request.query)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error generating embedding")

    try:
          results = store.query(query_embeddings=[query_embedding])
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error querying database")

    ids = [id_ for id_, distance in zip(results['ids'][0], results['distances'][0]) if distance <= request.threshold]

    return JSONResponse({"results": ids})


@app.post("/api/search/docs")
async def search_documents(request: TextQueryRequest):
    return await _text_query(request, text_store)


def _filter(items: list[str], image_store: chromadb.Collection | None = None, text_store: chromadb.Collection| None = None,video_store: chromadb.Collection| None = None ) -> list[str]:
        text_ids = _get_exisiting_ids(text_store)
        exclude = set(text_ids)
        return [item for item in items if item not in exclude]
  
def _get_exisiting_ids (store: chromadb.Collection| None = None) -> list[str]:
        limit = 100
        offset = 0
        ids = []
        if not store:
             return ids
        
        while True:
            batch = store.get(limit=limit, offset=offset)
            if not batch['ids']:
                break
            ids.extend(batch['ids'])
            offset += limit
        return ids


async def _index( ws: WebSocket, allowed_exts: tuple[str], indexer: DocIndexer, image_store: chromadb.Collection | None = None, text_store: chromadb.Collection| None = None,video_store: chromadb.Collection| None = None):
    msg = await ws.receive_json()
    if msg.get("action") == "index":
        dirpaths = msg.get("dirs", [])
        files = get_files_from_dirs(dirpaths, allowed_exts=allowed_exts)
        filtered_files = _filter(files, image_store, text_store, video_store)
        await indexer.run(filtered_files)
        await ws.close()
    else: 
        await ws.send_json(FailMessage(error="invalid action").model_dump())
        await ws.close()

@app.websocket("/ws/index/docs")
async def index(ws: WebSocket):
    await ws.accept()

    listener = FileIndexerWebSocketListener(ws,store=text_store)
    indexer = DocIndexer(text_encoder=text_encoder,listener=listener)

    try:
        await _index(ws, SupportedFileTypes.TEXT, indexer, text_store=text_store)
    except RuntimeError:
         print("Runtime Error")
    except WebSocketDisconnect:
        print("Client disconnected")


async def _count(store: chromadb.Collection):
    try:
        count = await run_in_threadpool(store.count)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"count": count})

@app.get("/api/count/docs")
async def count_documents_collection():
    return await _count(text_store)
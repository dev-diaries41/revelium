from fastapi import FastAPI, HTTPException,  WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from api.indexer import PromptIndexerWebSocketListener
from api.indexer import PromptIndexerWebSocketListener, FailMessage
from revelium.core.engine import Revelium

revelium = Revelium(indexer_listener=PromptIndexerWebSocketListener())

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

@app.websocket("/ws/index/")
async def index(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        if msg.get("action") == "index":
            # dirpaths = msg.get("dirs", [])
            # files = get_files_from_dirs(dirpaths, allowed_exts=allowed_exts)
            # filtered_files = _filter(files, revelium.prompt_embedding_store)
            # await revelium.indexer.run(filtered_files)
            await ws.close()
        else: 
            await ws.send_json(FailMessage(error="invalid action").model_dump())
            await ws.close()
    except RuntimeError:
         print("Runtime Error")
    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/api/count/prompts")
async def count_documents_collection():
    try:
        count = await revelium.prompt_store.count()
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"count": count})


@app.get("/api/count/clusters")
async def count_documents_collection():
    try:
        count = await revelium.cluster_embedding_store.count()
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"count": count})
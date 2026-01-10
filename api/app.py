import os 

from fastapi import FastAPI, HTTPException,  WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi import  UploadFile, Body
import json

from typing import List
from pydantic import BaseModel, Field
from api.indexer import BaseWebSocketListener, FailMessage
from revelium.core.engine import Revelium, ReveliumConfig
from revelium.prompts.types import Prompt

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

revelium = Revelium(config=ReveliumConfig(provider_api_key=OPENAI_API_KEY))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

# @app.websocket("/ws/promtps/index")
# async def index_prompts(ws: WebSocket):
#     await ws.accept()
#     try:
#         msg = await ws.receive_json()
#         if msg.get("action") == "index":
#             # TODO:  retreive prompts
#             # await revelium.index(prompts)
#             await ws.close()
#         else: 
#             await ws.send_json(FailMessage(error="invalid action").model_dump())
#             await ws.close()
#     except RuntimeError:
#          print("Runtime Error")
#     except WebSocketDisconnect:
#         print("Client disconnected")


class AddPromptsRequest(BaseModel):
    prompts: List[Prompt]


@app.post("/api/prompts/add")
async def add_prompts(req: AddPromptsRequest):
    try:
        if len(req.prompts) == 0:
            raise (HTTPException(status_code=400, detail="Missing prompts"))
        
        # await revelium.prompt_embedding_store.add(req.prompts)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"prompts_added": len(req.prompts)})



@app.get("/api/prompts/count")
async def count_prompts():
    try:
        count = await run_in_threadpool( revelium.prompt_embedding_store.count)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"count": count})


@app.get("/api/clusters/count")
async def count_clusters():
    try:
        count = await run_in_threadpool( revelium.cluster_embedding_store.count)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"count": count})
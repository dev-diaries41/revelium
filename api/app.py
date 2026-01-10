import os 

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from typing import List
from dataclasses import asdict
from pydantic import BaseModel, Field
from revelium.core.engine import Revelium, ReveliumConfig
from revelium.prompts.types import Prompt

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

revelium = Revelium(config=ReveliumConfig(provider_api_key=OPENAI_API_KEY))
revelium.text_embedder.init()

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
            raise HTTPException(status_code=400, detail="Missing prompts")
        # Note: This approach my be temp
        result = await revelium.index_prompts(req.prompts)
        if hasattr(result, "error" ):
             raise result.error
        result_dict = {k: v for k, v in asdict(result).items() if k != "error"}
    except Exception as e:
            raise HTTPException(status_code=500, detail="Error adding prompts")
    return JSONResponse(result_dict)



class GetPromptsRequest(BaseModel):
    prompt_ids: List[str]


@app.post("/api/prompts/")
async def get_prompts(req: GetPromptsRequest):
    try:
        if len(req.prompt_ids) == 0:
            raise HTTPException(status_code=400, detail="Missing prompt ids")
        prompts = await run_in_threadpool(revelium.get_prompts_by_ids, req.prompt_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse({"prompts": [asdict(p) for p in prompts]})


@app.get("/api/prompts/count")
async def count_prompts():
    try:
        count = await run_in_threadpool( revelium.prompt_embedding_store.count)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"count": count})


@app.get("/api/clusters/metadata")
async def get_cluster_metadata(cluster_id: str):
    try:
        metadata = await run_in_threadpool(revelium.get_cluster_metadata, cluster_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse({"metadata": metadata})


@app.get("/api/clusters/count")
async def count_clusters():
    try:
        count = await run_in_threadpool( revelium.cluster_embedding_store.count)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse({"count": count})

@app.get("/api/labels")
async def get_existing_labels():
    try:
        labels = await run_in_threadpool(revelium.get_existing_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse({"labels": labels})

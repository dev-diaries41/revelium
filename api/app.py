import os 
import json 

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from pydantic import ValidationError
from typing import List, Optional
from dataclasses import asdict
from smartscan.processor.metrics import MetricsSuccess

from revelium.prompts.types import Prompt
from revelium.core.engine import Revelium, ReveliumConfig
from revelium.schemas.api import AddPromptsRequest, GetPromptsRequest, GetPromptsResponse, GetCountResponse, GetLabelsResponse, GetClustersResponse, GetPromptsOverviewResponse, UpdateLabelResponse

from dotenv import load_dotenv
from revelium.constants.api import ADD_PROMPTS_ENDPOINT, ADD_PROMPTS_FILE_ENDPOINT, BASE_PROMPTS_ENDPOINT, GET_PROMPTS_OVERVIEW_ENDPOINT, GET_CLUSTER_LABELS_ENDPOINT, COUNT_CLUSTERS_ENDPOINT, COUNT_PROMPTS_ENDPOINT, BASE_CLUSTER_ENDPOINT, START_CLUSTERING_ENDPOINT
from revelium.constants.llms import OPENAI_API_KEY

load_dotenv()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

revelium = Revelium(config=ReveliumConfig(provider_api_key=OPENAI_API_KEY))
revelium.text_embedder.init()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["POST", "OPTIONS", "GET", "PATCH"],
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




@app.post(ADD_PROMPTS_ENDPOINT)
async def add_prompts(req: AddPromptsRequest):
    try:
        if len(req.prompts) == 0:
            raise HTTPException(status_code=400, detail="Missing prompts")
        # TODO: Add job to queue and return JobReceipt
        result = await revelium.index_prompts(req.prompts)
        if hasattr(result, "error" ):
            raise result.error
        result_dict: MetricsSuccess = {k: v for k, v in asdict(result).items() if k != "error"}
    except Exception as e:
            raise HTTPException(status_code=500, detail="Error adding prompts")
    return JSONResponse(result_dict)



@app.post(ADD_PROMPTS_FILE_ENDPOINT)
async def add_prompts_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        prompts_data = json.loads(content)

        # Validate prompts
        prompts: List[Prompt] = [Prompt(**p) for p in prompts_data or  []]

        if not prompts or len(prompts) == 0:
            raise HTTPException(status_code=400, detail="Missing prompts in file")
        
        try:
            prompts: List[Prompt] = [Prompt(**p) for p in prompts_data or []]
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid prompts: {ve.errors()}")

        if not prompts or len(prompts) == 0:
            raise HTTPException(status_code=400, detail="Missing prompts in file")


        # TODO: Add job to queue and return JobReceipt
        result = await revelium.index_prompts(prompts)
        if hasattr(result, "error"):
            raise result.error

        result_dict: MetricsSuccess = {k: v for k, v in asdict(result).items()}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse(result_dict)


@app.post(BASE_PROMPTS_ENDPOINT)
async def get_prompts(req: GetPromptsRequest):
    try:
        if len(req.prompt_ids) == 0:
            raise HTTPException(status_code=400, detail="Missing prompt ids")
        prompts = await run_in_threadpool(revelium.get_prompts_by_ids, req.prompt_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse(GetPromptsResponse(prompts=prompts).model_dump())


@app.get(COUNT_PROMPTS_ENDPOINT)
async def count_prompts():
    try:
        count = await run_in_threadpool( revelium.prompt_embedding_store.count)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse(GetCountResponse(count=count).model_dump())


@app.get(GET_PROMPTS_OVERVIEW_ENDPOINT)
async def get_prompts_overview():
    try:
        overview = await run_in_threadpool( revelium.get_prompts_overview)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error getting prompt overview")
    return JSONResponse(GetPromptsOverviewResponse(**overview.model_dump()).model_dump())


@app.post(START_CLUSTERING_ENDPOINT)
async def cluster_prompts():
    try:
        res = await run_in_threadpool(revelium.cluster_prompts)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse({"status": "in_progress"}) # testing only

@app.get(BASE_CLUSTER_ENDPOINT)
async def get_clusters(cluster_id: Optional[str] = Query(None), limit: Optional[str] = Query(None), offset: Optional[str] = Query(None)):
    try:
        limit_int = int(limit) if limit not in (None, "") else None
        offset_int = int(offset) if offset not in (None, "") else None

        clusters = await run_in_threadpool(revelium.get_clusters, cluster_id, limit_int, offset_int, ['metadatas'])
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse(GetClustersResponse(clusters=list(clusters.values())).model_dump())


@app.patch(BASE_CLUSTER_ENDPOINT)
async def get_clusters(cluster_id: str, label: str):
    try:
        updated = await run_in_threadpool(revelium.update_cluster_label, cluster_id, label)
        if not updated:
            raise ValueError("Label not updated")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse(UpdateLabelResponse(updated_label=label).model_dump())


@app.get(COUNT_CLUSTERS_ENDPOINT)
async def count_clusters():
    try:
        count = await run_in_threadpool( revelium.cluster_embedding_store.count)
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error counting items in collection")
    return JSONResponse(GetCountResponse(count=count).model_dump())

@app.get(GET_CLUSTER_LABELS_ENDPOINT)
async def get_existing_labels():
    try:
        labels = await run_in_threadpool(revelium.get_existing_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse(GetLabelsResponse(labels=labels).model_dump())

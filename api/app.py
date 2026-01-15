import json 

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from pydantic import ValidationError
from typing import List, Optional
from dataclasses import asdict

from smartscan.processor.metrics import MetricsSuccess

from revelium.errors import ReveliumError, ErrorCode
from revelium.schemas.llm import LLMClientConfig
from revelium.prompts.types import Prompt
from revelium.prompts.prompts_manager import PromptsManager
from revelium.schemas.api import AddPromptsRequest, GetPromptsRequest, GetPromptsResponse, GetCountResponse, GetLabelsResponse, GetClustersResponse, GetPromptsOverviewResponse, UpdateLabelResponse, GetClustersAccuracyResponse, QueryPromptsRequest, UpdatePromptClusterIdResponse
from revelium.constants.api import Routes
from revelium.constants.models import OPENAI_API_KEY, DEFAULT_SYSTEM_PROMPT, DEFAULT_OPENAI_MODEL
from revelium.constants import DEFAULT_CHROMADB_PATH
from revelium.prompts.cluster import cluster_prompts, get_cluster_plot
from revelium.models.manage import ModelManager
from revelium.embeddings.helpers import get_embedding_store
from revelium.prompts.indexer import PromptIndexer
from revelium.prompts.indexer_listener import PromptIndexListener
from revelium.providers.llm.openai import OpenAIClient
from dotenv import load_dotenv

load_dotenv()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

model_manager = ModelManager()
text_embedder = model_manager.get_text_embedder('all-minilm-l6-v2')
text_embedder.init()
prompt_embedding_store =  get_embedding_store(DEFAULT_CHROMADB_PATH, PromptsManager.PROMPT_TYPE, 'all-minilm-l6-v2', text_embedder.embedding_dim) 
cluster_embedding_store =  get_embedding_store(DEFAULT_CHROMADB_PATH, PromptsManager.CLUSTER_TYPE, 'all-minilm-l6-v2', text_embedder.embedding_dim) 
llm = OpenAIClient(OPENAI_API_KEY, LLMClientConfig(model_name=DEFAULT_OPENAI_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT))
prompts_manager = PromptsManager(llm_client=llm, prompt_embedding_store=prompt_embedding_store, cluster_embedding_store=cluster_embedding_store)
indexer =  PromptIndexer(text_embedder, listener=PromptIndexListener(prompts_manager =prompts_manager), embeddings_store=prompt_embedding_store, batch_size=100, max_concurrency=4)


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

@app.post(Routes.ADD_PROMPTS_ENDPOINT)
async def add_prompts(req: AddPromptsRequest):
    if not req.prompts:
        raise HTTPException(status_code=400, detail="Missing prompts")
    # TODO: Add job to queue and return JobReceipt
    result = await indexer.run(req.prompts)
    if hasattr(result, "error" ):
        raise result.error
    result_dict: MetricsSuccess = {k: v for k, v in asdict(result).items() if k != "error"}
    return JSONResponse(result_dict)



@app.post(Routes.ADD_PROMPTS_FILE_ENDPOINT)
async def add_prompts_file(file: UploadFile = File(...)):
    content = await file.read()
    prompts_data = json.loads(content)

    try:
        prompts: List[Prompt] = [Prompt(**p) for p in prompts_data or []]
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid prompts: {ve.errors()}")

    if not prompts or len(prompts) == 0:
        raise HTTPException(status_code=400, detail="Missing prompts in file")

    # TODO: Add job to queue and return JobReceipt
    result = await indexer.run(prompts)
    if hasattr(result, "error"):
        raise result.error

    result_dict: MetricsSuccess = {k: v for k, v in asdict(result).items()}

    return JSONResponse(result_dict)


@app.post(Routes.BASE_PROMPTS_ENDPOINT)
async def get_prompts(req: GetPromptsRequest):
    if req.prompt_ids and req.limit:
        raise HTTPException(status_code=400, detail="Prompt ids and limit filtering must be used seperately")
    prompts = await run_in_threadpool(prompts_manager.get_prompts_paginate, req.prompt_ids, req.cluster_id, req.limit, req.offset)
    return JSONResponse(GetPromptsResponse(prompts=prompts).model_dump())


@app.post(Routes.QUERY_PROMPTS_ENDPOINT)
async def query_prompts(req: QueryPromptsRequest):
    prompts = await run_in_threadpool(prompts_manager.query_prompts, text_embedder, req.query, req.cluster_id, req.limit)
    return JSONResponse(GetPromptsResponse(prompts=prompts).model_dump())


@app.get(Routes.COUNT_PROMPTS_ENDPOINT)
async def count_prompts():
    count = await run_in_threadpool( prompts_manager.prompt_embedding_store.count)
    return JSONResponse(GetCountResponse(count=count).model_dump())


@app.get(Routes.GET_PROMPTS_OVERVIEW_ENDPOINT)
async def get_prompts_overview():
    overview = await run_in_threadpool( prompts_manager.get_prompts_overview)
    return JSONResponse(GetPromptsOverviewResponse(**overview.model_dump()).model_dump())


@app.patch(Routes.BASE_PROMPTS_ENDPOINT)
async def update_prompt_cluster(prompt_id: str, cluster_id: str):
    try:
        await run_in_threadpool( prompts_manager.update_prompts , {prompt_id : cluster_id}, {})
    except ReveliumError as e:
        if e.code == ErrorCode.PROMPT_NOT_FOUND:
            raise HTTPException(status_code=404, detail="Prompt not found")
        else:
            raise e
    return JSONResponse(UpdatePromptClusterIdResponse(updated_cluster_id=cluster_id).model_dump())




@app.post(Routes.START_CLUSTERING_ENDPOINT)
async def start_clustering_prompts():
    _ = await run_in_threadpool(cluster_prompts, prompts_manager)
    return JSONResponse({"status": "in_progress"}) # testing only

@app.get(Routes.BASE_CLUSTER_ENDPOINT)
async def get_clusters(cluster_id: Optional[str] = Query(None), limit: Optional[str] = Query(None), offset: Optional[str] = Query(None)):
    limit_int = int(limit) if limit not in (None, "") else None
    offset_int = int(offset) if offset not in (None, "") else None
    clusters = await run_in_threadpool(prompts_manager.get_clusters, cluster_id, limit_int, offset_int, ['metadatas'])
    return JSONResponse(GetClustersResponse(clusters=list(clusters.values())).model_dump())


@app.patch(Routes.BASE_CLUSTER_ENDPOINT)
async def update_cluster_label(cluster_id: str, label: str):
    updated = await run_in_threadpool(
        prompts_manager.update_cluster_label, cluster_id, label
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Label doesn't exist")

    return JSONResponse(UpdateLabelResponse(updated_label=label).model_dump())


@app.get(Routes.COUNT_CLUSTERS_ENDPOINT)
async def count_clusters():
    count = await run_in_threadpool( prompts_manager.cluster_embedding_store.count)
    return JSONResponse(GetCountResponse(count=count).model_dump())

@app.get(Routes.GET_CLUSTER_LABELS_ENDPOINT)
async def get_existing_labels():
    labels = await run_in_threadpool(prompts_manager.get_existing_labels)
    return JSONResponse(GetLabelsResponse(labels=labels).model_dump())


@app.get(Routes.GET_CLUSTER_ACCURACY_ENDPOINT)
async def get_cluster_accuracy():
    accuracy = await run_in_threadpool(prompts_manager.calculate_cluster_accuracy)
    return JSONResponse(GetClustersAccuracyResponse(accuracy=accuracy).model_dump())


@app.get(Routes.GET_CLUSTER_PLOT_ENDPOINT)
async def get_clusters_plot():
    img_bytes = await run_in_threadpool(get_cluster_plot, prompts_manager)
    return Response(status_code=200 if img_bytes else 204, content=img_bytes, media_type="image/png")

from typing import List, Optional
import httpx

from smartscan import ClusterAccuracy
from revelium.constants.api import ADD_PROMPTS_ENDPOINT, ADD_PROMPTS_FILE_ENDPOINT, BASE_PROMPTS_ENDPOINT, GET_PROMPTS_OVERVIEW_ENDPOINT, GET_CLUSTER_LABELS_ENDPOINT, COUNT_CLUSTERS_ENDPOINT, COUNT_PROMPTS_ENDPOINT, BASE_CLUSTER_ENDPOINT, START_CLUSTERING_ENDPOINT, GET_CLUSTER_ACCURACY_ENDPOINT, QUERY_PROMPTS_ENDPOINT, GET_CLUSTER_PLOT_ENDPOINT
from revelium.prompts.types import Prompt, PromptsOverviewInfo
from revelium.schemas.api import AddPromptsRequest, GetPromptsRequest, GetClusterRequestParams, ClusterNoEmbeddings, UpdateLabelParams, QueryPromptsRequest

class ReveliumClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")


    async def add_prompts(self, prompts: List[Prompt]) -> dict:
        url = f"{self.base_url}{ADD_PROMPTS_ENDPOINT}"
        # Convert dataclasses to dicts
        # TODD: chane prompts to pydantic basemodel
        payload = AddPromptsRequest(prompts=prompts)

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload.model_dump())
            res.raise_for_status() 
            return res.json()
        
    
    async def add_prompts_file(self, file_path: str) -> dict:
        """
        Upload a JSON file containing prompts.
        """
        url = f"{self.base_url}{ADD_PROMPTS_FILE_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": (file_path, f, "application/json")}
                res = await client.post(url, files=files)
            res.raise_for_status() 
            return res.json()
        
    async def get_prompts(self, ids: Optional[List[str]] = None, cluster_id: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Prompt]:
        url = f"{self.base_url}{BASE_PROMPTS_ENDPOINT}"
        payload = GetPromptsRequest(prompt_ids=ids, cluster_id=cluster_id, limit=limit, offset=offset)

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload.model_dump())
            res.raise_for_status() 
            return res.json().get('prompts', [])
        
    async def query_prompts(self, query: str, cluster_id: Optional[str] = None, limit: Optional[int] = None) -> List[Prompt]:
        url = f"{self.base_url}{QUERY_PROMPTS_ENDPOINT}"
        payload = QueryPromptsRequest(query=query, cluster_id=cluster_id, limit=limit)

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload.model_dump())
            res.raise_for_status() 
            return res.json().get('prompts', [])
        
        
    async def get_prompts_overview(self) -> PromptsOverviewInfo:
        url = f"{self.base_url}{GET_PROMPTS_OVERVIEW_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            res.raise_for_status() 
            return res.json()


    ## NOTE: May have to queue and return job id
    async def cluster_prompts(self) -> dict:
        url = f"{self.base_url}{START_CLUSTERING_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.post(url)
            res.raise_for_status() 
            return res.json()
        

    async def get_clusters(self, cluster_id: Optional[str] = None, limit: Optional[str] = None, offset: Optional[str] = None ) ->List[ClusterNoEmbeddings]:
        url = f"{self.base_url}{BASE_CLUSTER_ENDPOINT}"
        params = GetClusterRequestParams(cluster_id=cluster_id, limit=limit, offset=offset)

        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params.model_dump())
            res.raise_for_status() 
            return res.json().get("clusters", [])
   

    async def update_cluster_label(self, cluster_id: str, label: str) -> str:
        url = f"{self.base_url}{BASE_CLUSTER_ENDPOINT}"
        params = UpdateLabelParams(cluster_id=cluster_id, label=label)

        async with httpx.AsyncClient() as client:
            res = await client.patch(url, params=params.model_dump())
            res.raise_for_status() 
            return res.json().get("updated_label")
        
    async def count_prompts(self) -> int:
        url = f"{self.base_url}{COUNT_PROMPTS_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            res.raise_for_status() 
            return res.json().get("count", 0)

   
    async def count_clusters(self) -> int:
        url = f"{self.base_url}{COUNT_CLUSTERS_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            res.raise_for_status() 
            return res.json().get("count", 0)

    async def get_existing_labels(self) -> list[str]:
        url = f"{self.base_url}{GET_CLUSTER_LABELS_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            res.raise_for_status() 
            return res.json().get("labels", [])
        
    async def get_cluster_accuracy(self) -> ClusterAccuracy:
        url = f"{self.base_url}{GET_CLUSTER_ACCURACY_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            res.raise_for_status() 
            return res.json().get("accuracy")
        
    async def get_cluster_plot(self) -> bytes:
        """
        Fetches the cluster plot image as PNG bytes.
        """
        url = f"{self.base_url}{GET_CLUSTER_PLOT_ENDPOINT}"
        # params = {"method": method}

        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            res.raise_for_status()
            return res.content 
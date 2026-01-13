from typing import List, Optional
import httpx

from revelium.prompts.types import Prompt, PromptsOverviewInfo
from revelium.schemas.api import AddPromptsRequest, GetPromptsRequest, GetClusterRequestParams, ClusterNoEmbeddings, UpdateLabelParams

class ReveliumClient:
    ADD_PROMPTS_ENDPOINT = "/api/prompts/add"
    ADD_PROMPTS_FILE_ENDPOINT = "/api/prompts/add/file"
    GET_PROMPTS_ENDPOINT = "/api/prompts"
    GET_PROMPTS_OVERVIEW_ENDPOINT = "/api/prompts/overview"
    COUNT_PROMPTS_ENDPOINT = "/api/prompts/count"
    BASE_CLUSTER_ENDPOINT = "/api/clusters"
    START_CLUSTERING_ENDPOINT = f"{BASE_CLUSTER_ENDPOINT}/start"
    COUNT_CLUSTERS_ENDPOINT = "/api/clusters/count"
    GET_LABELS_ENDPOINT = "/api/labels"
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")


    async def add_prompts(self, prompts: List[Prompt]) -> dict:
        url = f"{self.base_url}{ReveliumClient.ADD_PROMPTS_ENDPOINT}"
        # Convert dataclasses to dicts
        # TODD: chane prompts to pydantic basemodel
        payload = AddPromptsRequest(prompts=prompts)

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload.model_dump())
            if res.status_code != 200:
                raise Exception(f"Error adding prompts: {res.text}")
            return res.json()
        
    
    async def add_prompts_file(self, file_path: str) -> dict:
        """
        Upload a JSON file containing prompts.
        """
        url = f"{self.base_url}{ReveliumClient.ADD_PROMPTS_FILE_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": (file_path, f, "application/json")}
                res = await client.post(url, files=files)
            if res.status_code != 200:
                raise Exception(f"Error adding prompts from file: {res.text}")
            return res.json()
        
    async def get_prompts(self, ids: List[str]) -> List[Prompt]:
        url = f"{self.base_url}{ReveliumClient.GET_PROMPTS_ENDPOINT}"
        payload = GetPromptsRequest(prompt_ids=ids)

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload.model_dump())
            if res.status_code != 200:
                raise Exception(f"Error adding prompts: {res.text}")
            return res.json().get('prompts', [])
        
    async def get_prompts_overview(self) -> PromptsOverviewInfo:
        url = f"{self.base_url}{ReveliumClient.GET_PROMPTS_OVERVIEW_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error getting ovrerview: {res.text}")
            return res.json()


    ## NOTE: May have to queue and return job id
    async def cluster_prompts(self) -> dict:
        url = f"{self.base_url}{ReveliumClient.START_CLUSTERING_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.post(url)
            if res.status_code != 200:
                raise Exception(f"Error clustering prompts: {res}")
            return res.json()
        

    async def get_clusters(self, cluster_id: Optional[str] = None, limit: Optional[str] = None, offset: Optional[str] = None ) ->List[ClusterNoEmbeddings]:
        url = f"{self.base_url}{ReveliumClient.BASE_CLUSTER_ENDPOINT}"
        params = GetClusterRequestParams(cluster_id=cluster_id, limit=limit, offset=offset)

        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params.model_dump())
            if res.status_code != 200:
                raise Exception(f"Error getting clusters: {res.text}")
            return res.json().get("clusters", [])
   

    async def update_cluster_label(self, cluster_id: str, label: str) -> str:
        url = f"{self.base_url}{ReveliumClient.BASE_CLUSTER_ENDPOINT}"
        params = UpdateLabelParams(cluster_id=cluster_id, label=label)

        async with httpx.AsyncClient() as client:
            res = await client.patch(url, params=params.model_dump())
            if res.status_code != 200:
                raise Exception(f"Error updating label: {res.text}")
            return res.json().get("updated_label")
        
    async def count_prompts(self) -> int:
        url = f"{self.base_url}{ReveliumClient.COUNT_PROMPTS_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error counting prompts: {res.text}")
            return res.json().get("count", 0)

   
    async def count_clusters(self) -> int:
        url = f"{self.base_url}{ReveliumClient.COUNT_CLUSTERS_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error counting clusters: {res.text}")
            return res.json().get("count", 0)

    async def get_existing_labels(self) -> list[str]:
        url = f"{self.base_url}{ReveliumClient.GET_LABELS_ENDPOINT}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error getting labels: {res.text}")
            return res.json().get("labels", [])
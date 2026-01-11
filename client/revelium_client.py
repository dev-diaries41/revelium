from typing import List, Optional
import httpx

from revelium.prompts.types import Prompt, PromptsOverviewInfo
from smartscan import ClusterMetadata

class ReveliumClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")


    async def add_prompts(self, prompts: List[Prompt]) -> dict:
        url = f"{self.base_url}/api/prompts/add"
        # Convert dataclasses to dicts
        # TODD: chane prompts to pydantic basemodel
        payload = {"prompts": [p.model_dump() for p in prompts]}

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload)
            if res.status_code != 200:
                raise Exception(f"Error adding prompts: {res.text}")
            return res.json()
        
    async def get_prompts(self, ids: List[str]) -> List[Prompt]:
        url = f"{self.base_url}/api/prompts/"
        payload = {"prompt_ids": ids}

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload)
            if res.status_code != 200:
                raise Exception(f"Error adding prompts: {res.text}")
            return res.json().get('prompts', [])
        
    async def get_prompts_overview(self) -> PromptsOverviewInfo:
        url = f"{self.base_url}/api/prompts/overview"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error getting ovrerview: {res.text}")
            return res.json()


    async def get_cluster_metadata(self, cluster_id: str) -> Optional[ClusterMetadata]:
        url = f"{self.base_url}/api/clusters/metadata"
        payload = {"cluster_id": cluster_id}

        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=payload)
            if res.status_code != 200:
                raise Exception(f"Error getting metadata: {res.text}")
            return res.json().get("metadata")

   
    async def count_prompts(self) -> int:
        url = f"{self.base_url}/api/prompts/count"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error counting prompts: {res.text}")
            return res.json().get("count", 0)

   
    async def count_clusters(self) -> int:
        url = f"{self.base_url}/api/clusters/count"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error counting clusters: {res.text}")
            return res.json().get("count", 0)

    async def get_existing_labels(self) -> list[str]:
        url = f"{self.base_url}/api/labels"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise Exception(f"Error getting labels: {res.text}")
            return res.json().get("labels")
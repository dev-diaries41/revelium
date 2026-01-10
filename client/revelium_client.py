from typing import List
import httpx
from revelium.prompts.types import Prompt
from dataclasses import asdict

class ReveliumClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")


    async def add_prompts(self, prompts: List[Prompt]) -> dict:
        """
        Send prompts to the POST /api/prompts/add endpoint.
        """
        url = f"{self.base_url}/api/prompts/add"
        # Convert dataclasses to dicts
        payload = {"prompts": [asdict(p) for p in prompts]}

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise Exception(f"Error adding prompts: {resp.text}")
            return resp.json()

   
    async def count_prompts(self) -> int:
        url = f"{self.base_url}/api/prompts/count"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise Exception(f"Error counting prompts: {resp.text}")
            return resp.json().get("count", 0)

   
    async def count_clusters(self) -> int:
        url = f"{self.base_url}/api/clusters/count"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise Exception(f"Error counting clusters: {resp.text}")
            return resp.json().get("count", 0)

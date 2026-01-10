import asyncio
from revelium.prompts.types import Prompt
from client.revelium_client import ReveliumClient

async def main():
    client = ReveliumClient("http://localhost:8000")

    # Add prompts via JSON body
    prompts = [
        Prompt("prompt_test_1", "Summarize Q4 sales"),
        Prompt("prompt_test_2", "Summarize Q2 sales"),
    ]
    print(await client.add_prompts(prompts=prompts))

    # Count prompts
    print("Total prompts:", await client.count_prompts())

    # Count clusters
    print("Total clusters:", await client.count_clusters())

asyncio.run(main())

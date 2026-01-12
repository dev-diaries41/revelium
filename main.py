import asyncio
from revelium.prompts.types import Prompt
from client.revelium_client import ReveliumClient

async def main():
    client = ReveliumClient("http://127.0.0.1:8000")

    # Add prompts via JSON body
    prompts = [
        Prompt(prompt_id = "prompt_test_1", content="Summarize Q4 sales"),
        Prompt(prompt_id = "prompt_test_2", content="Summarize Q2 sales"),
    ]
    print(await client.add_prompts(prompts=prompts))

    # Count prompts
    print("Total prompts:", await client.count_prompts())

    # Count clusters
    print("Total clusters:", await client.count_clusters())

    print("Cluster metadata:", await client.get_cluster_metadata("fce4cfdc44b3ea3f"))

    print("Prompts:", await client.get_prompts([p.prompt_id for p in prompts]))

    print("Labels:", await client.get_existing_labels())

    print("Overview:", await client.get_prompts_overview())


asyncio.run(main())

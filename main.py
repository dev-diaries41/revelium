import asyncio
import json
from revelium.prompts.types import Prompt
from client.revelium_client import ReveliumClient
from revelium.data import get_dummy_data

async def main():
    client = ReveliumClient("http://127.0.0.1:8000")

    # Add prompts via JSON body
    prompts = get_dummy_data(n=5)
    # print(prompts)
    # with open("dummy_prompts.json","w") as f:
    #     json.dump(prompts, f, indent=1)
    #     print(await client.add_prompts(prompts=prompts))

    # Count prompts
    print("Total prompts:", await client.count_prompts())

    # Count clusters
    print("Total clusters:", await client.count_clusters())

    print("Clusters:", await client.get_clusters("fce4cfdc44b3ea3f"))

    print("Clustering:", await client.cluster_prompts())
    
    # print("Prompts:", await client.get_prompts([p.get("prompt_id") if isinstance(p, dict) else p.prompt_id for p in prompts]))

    # print("Labels:", await client.get_existing_labels())
    print("Update label:", await client.update_cluster_label(cluster_id="0173bc6fc4524fcaaee3f165410704f7", label="test label"))

    print("Overview:", await client.get_prompts_overview())

    # print("Add prompts file:", await client.add_prompts_file("output/dummy_prompts.json"))

asyncio.run(main())

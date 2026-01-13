import pytest
import pytest_asyncio
from revelium.data import get_dummy_data
from client.revelium_client import ReveliumClient
from revelium.prompts.types import Prompt

@pytest_asyncio.fixture
async def setup_client():
    client = ReveliumClient("http://127.0.0.1:8000")
    prompts = get_dummy_data(n=5)
    return client, prompts

@pytest.mark.asyncio
class TestReveliumClient:

    async def test_count_prompts(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        total_prompts = await client.count_prompts()
        assert isinstance(total_prompts, int)

    async def test_count_clusters(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        total_clusters = await client.count_clusters()
        assert isinstance(total_clusters, int)

    async def test_get_clusters(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        clusters = await client.get_clusters("fce4cfdc44b3ea3f")  # Replace with valid prompt_id
        assert isinstance(clusters, list)

    async def test_cluster_prompts(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        clustering_result = await client.cluster_prompts()
        assert clustering_result is not None

    async def test_get_prompts(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, prompts = setup_client
        prompt_ids = [p.get("prompt_id") if isinstance(p, dict) else p.prompt_id for p in prompts]
        retrieved_prompts = await client.get_prompts(prompt_ids)
        assert isinstance(retrieved_prompts, list)

    async def test_labels(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        labels = await client.get_existing_labels()
        assert isinstance(labels, list)

    async def test_update_cluster_label(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        label = "test label"
        update_result = await client.update_cluster_label(
            cluster_id="0173bc6fc4524fcaaee3f165410704f7",  # Replace with valid cluster_id
            label="test label"
        )
        assert update_result == label

    async def test_prompts_overview(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        overview = await client.get_prompts_overview()
        assert isinstance(overview, dict)

    async def test_add_prompts_file(self, setup_client: tuple[ReveliumClient, list[Prompt]]):
        client, _ = setup_client
        add_file_result = await client.add_prompts_file("output/dummy_prompts.json")
        assert add_file_result is not None

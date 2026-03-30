import httpx
import pytest

from app.services.news.search import (
    BraveNewsSearchProvider,
    WebSearchResult,
)


@pytest.mark.asyncio
async def test_brave_search_provider_parses_web_results():
    recorded_headers = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        recorded_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave result title",
                            "url": "https://example.com/brave-result",
                            "description": "Brave result snippet",
                        }
                    ]
                }
            },
        )

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = BraveNewsSearchProvider(
            client,
            api_key="brave-key",
            base_url="https://api.search.brave.com/res/v1/web/search",
        )
        results = await provider.search("abc news", limit=2)

    assert results == [
        WebSearchResult(
            title="Brave result title",
            url="https://example.com/brave-result",
            snippet="Brave result snippet",
            domain="example.com",
        )
    ]
    assert recorded_headers["x-subscription-token"] == "brave-key"

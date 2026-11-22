"""
Unit tests for the Real-time Web Search Tool.
Tests DuckDuckGo/ddgs search, Tavily fallback, response formatting, query cleaning,
and offline error handling via deterministic mocks.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.tools.web_search import (
    web_search,
    perform_web_search,
    _clean_query,
    classify_source_authority,
    _format_search_results,
)


class TestWebSearchOperations:
    """Test web search execution, authority scoring, and formatting with mocked responses."""

    def test_clean_query_helper(self) -> None:
        assert _clean_query("Search for the latest population estimate of India.") == "the latest population estimate of India"
        assert _clean_query("Please find the GDP of Japan") == "the GDP of Japan"
        assert _clean_query("Tell me about quantum computing?") == "quantum computing"
        assert _clean_query("India population") == "India population"

    def test_classify_source_authority_tiers(self) -> None:
        # High authority TLDs & Organizations
        assert "High Authority" in classify_source_authority("https://www.census.gov/popclock")
        assert "High Authority" in classify_source_authority("https://data.gov.in/resource")
        assert "High Authority" in classify_source_authority("https://www.un.org/development/desa/pd")
        assert "High Authority" in classify_source_authority("https://data.worldbank.org/indicator")
        assert "High Authority" in classify_source_authority("https://ox.ac.uk/research")

        # Standard Web Sources
        assert classify_source_authority("https://random-tech-blog.com/article") == "Standard Web Source"
        assert classify_source_authority("https://news-sample.net/item") == "Standard Web Source"

        # Low Authority / User Generated
        assert "Low Authority" in classify_source_authority("https://www.reddit.com/r/india/comments/abc")
        assert "Low Authority" in classify_source_authority("https://www.quora.com/What-is-the-population-of-India")
        assert "Low Authority" in classify_source_authority("https://twitter.com/user/status/123")

    def test_authoritative_source_prioritization_and_sorting(self) -> None:
        """Verify that search results are sorted so High Authority sources appear before low authority sources."""
        raw_items = [
            {
                "title": "Random Forum Post on Population",
                "href": "https://www.reddit.com/r/demographics/comments/123",
                "body": "A user says India population is 1.5 billion.",
            },
            {
                "title": "United Nations Population Fund - India",
                "href": "https://www.unfpa.org/data/world-population/IN",
                "body": "UN official demographic projection: 1.45 billion.",
            },
            {
                "title": "Tech Blog Analysis",
                "href": "https://tech-observer.org/demographics",
                "body": "Independent demographic commentary.",
            },
        ]

        formatted = _format_search_results(raw_items)

        # High Authority UN source should be ranked first [1]
        lines = formatted.split("\n\n")
        assert "[1] United Nations Population Fund - India" in lines[0]
        assert "High Authority" in lines[0]

        # Reddit forum should be sorted last [3] and labeled as Low Authority
        assert "[3] Random Forum Post on Population" in lines[2]
        assert "Low Authority" in lines[2]

    @patch("app.tools.web_search.DDGS")
    def test_duckduckgo_search_success(self, mock_ddgs_cls, mock_config) -> None:
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.return_value = [
            {
                "title": "India Population 2026",
                "href": "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=IN",
                "body": "India is home to an estimated 1,481,547,296 people.",
            },
            {
                "title": "World Population Prospects Report",
                "href": "https://www.un.org/wpp",
                "body": "Demographic statistics across Asia.",
            },
        ]

        result = perform_web_search("India population", provider="duckduckgo")

        assert "India Population 2026" in result
        assert "data.worldbank.org" in result
        assert "1,481,547,296" in result
        assert "High Authority" in result
        assert "[1]" in result
        assert "[2]" in result

    @patch("app.tools.web_search.DDGS")
    def test_duckduckgo_empty_results(self, mock_ddgs_cls, mock_config) -> None:
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.return_value = []
        if hasattr(mock_instance, "news"):
            mock_instance.news.return_value = []

        result = perform_web_search("completely_unique_nonexistent_query_xyz", provider="duckduckgo")
        assert "No search results found" in result

    @patch("app.tools.web_search.DDGS")
    def test_duckduckgo_timeout_or_error(self, mock_ddgs_cls, mock_config) -> None:
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.side_effect = TimeoutError("Connection timed out after 6s")

        result = perform_web_search("query", provider="duckduckgo")
        assert "Search error (DuckDuckGo):" in result
        assert "timed out" in result

    @patch("app.tools.web_search.DDGS")
    def test_duckduckgo_conversational_fallback(self, mock_ddgs_cls, mock_config) -> None:
        """When cleaned query returns 0, verify fallback to raw query or news."""
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        # 1st call (cleaned) -> empty; 2nd call (raw) -> results
        mock_instance.text.side_effect = [
            [],
            [{"title": "Census Data", "href": "https://census.gov/data", "body": "Latest census figures"}],
        ]

        result = perform_web_search("Search for the latest population estimate of India.", provider="duckduckgo")
        assert "Census Data" in result
        assert "https://census.gov/data" in result

    def test_empty_query_error(self, mock_config) -> None:
        assert "Error: Empty search query" in perform_web_search("")
        assert "Error: Empty search query" in perform_web_search("   ")

    def test_tavily_fallback_when_no_api_key(self, mock_config) -> None:
        with patch("app.tools.web_search._search_duckduckgo") as mock_ddg:
            mock_ddg.return_value = "DuckDuckGo fallback result"
            result = perform_web_search("test query", provider="tavily")
            assert result == "DuckDuckGo fallback result"

    @patch("tavily.TavilyClient")
    def test_tavily_search_success_with_authority_sorting(self, mock_client_cls, mock_config) -> None:
        mock_config.tavily_api_key = "tvly-test-key-12345"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Quora Discussion on India",
                    "url": "https://www.quora.com/india-gdp",
                    "content": "User speculations on GDP.",
                },
                {
                    "title": "World Bank India Data",
                    "url": "https://data.worldbank.org/country/india",
                    "content": "Official World Bank economic indicators.",
                },
            ]
        }

        result = perform_web_search("India GDP", provider="tavily")
        # World Bank should be ranked first [1]
        lines = result.split("\n\n")
        assert "[1] World Bank India Data" in lines[0]
        assert "High Authority" in lines[0]
        assert "[2] Quora Discussion on India" in lines[1]
        assert "Low Authority" in lines[1]

    @patch("app.tools.web_search.DDGS")
    def test_langchain_tool_wrapper(self, mock_ddgs_cls, mock_config) -> None:
        mock_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_instance
        mock_instance.text.return_value = [
            {"title": "Result 1", "href": "https://example.com", "body": "Snippet 1"}
        ]

        result = web_search.invoke({"query": "LangChain agents"})
        assert "Result 1" in result
        assert web_search.name == "web_search"

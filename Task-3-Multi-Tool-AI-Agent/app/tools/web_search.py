"""
Real-time Web Search Tool for Task 3.
Provides DuckDuckGo search as the default provider and optional Tavily search.
Migrated to the maintained 'ddgs' package with graceful fallback to 'duckduckgo_search'.
Extracts titles, URLs, and snippets. Prioritizes authoritative sources and identifies low-credibility platforms.
Handles network errors, timeouts, rate limits, and conversational query noise.
Never fabricates search results.
"""

import re
import warnings
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

# Suppress legacy rename warnings if old package is invoked
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore
    try:
        from duckduckgo_search.exceptions import DuckDuckGoSearchException as DDGSException  # type: ignore
    except ImportError:
        DDGSException = Exception  # type: ignore

from langchain_core.tools import tool

from app.core.config import config
from app.core.logger import logger

# Authoritative domain classifications for ranking and provenance tagging
HIGH_AUTHORITY_DOMAINS = {
    # International & Multilateral Organizations
    "un.org", "unfpa.org", "worldbank.org", "who.int", "imf.org", "wto.org", "oecd.org",
    # Statistical agencies & primary repositories
    "census.gov", "data.gov", "nih.gov", "cdc.gov", "statista.com", "macrotrends.net",
    "worldometers.info", "worldometer.info", "ourworldindata.org",
    # Major Global News Agencies & Scientific Publications
    "reuters.com", "apnews.com", "bloomberg.com", "bbc.com", "bbc.co.uk",
    "nature.com", "science.org", "economist.com",
    # Encyclopedic Reference
    "wikipedia.org", "britannica.com",
}

LOW_AUTHORITY_DOMAINS = {
    "quora.com", "reddit.com", "pinterest.com", "answers.com", "yahoo.answers.com",
    "tiktok.com", "instagram.com", "facebook.com", "twitter.com", "x.com",
    "tumblr.com", "buzzfeed.com",
}


def classify_source_authority(url: str) -> str:
    """
    Classify the credibility tier of a given source URL.
    Returns clear tier labeling: High Authority, Standard Web Source, or Low Authority.
    """
    try:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return "Standard Web Source"

        # Check governmental, academic, military, and international TLDs
        if any(
            netloc.endswith(tld)
            for tld in [".gov", ".gov.in", ".gov.uk", ".gov.au", ".edu", ".edu.in", ".ac.uk", ".mil", ".int"]
        ):
            return "High Authority (Governmental/Institutional/Educational)"

        # Check known authoritative statistical and news domains
        for domain in HIGH_AUTHORITY_DOMAINS:
            if domain in netloc:
                return "High Authority (Official/Statistical/Verified Reference)"

        # Check user-generated and forum domains
        for domain in LOW_AUTHORITY_DOMAINS:
            if domain in netloc:
                return "Low Authority (User-Generated/Forum/Social - Verify with primary source)"

        return "Standard Web Source"
    except Exception:
        return "Standard Web Source"


def _authority_score(url: str) -> int:
    """Numerical score for sorting: High Authority (2) > Standard (1) > Low Authority (0)."""
    classification = classify_source_authority(url)
    if "High Authority" in classification:
        return 2
    if "Low Authority" in classification:
        return 0
    return 1


def _clean_query(raw_query: str) -> str:
    """Strip conversational filler from search query to improve search engine recall."""
    cleaned = re.sub(
        r"^(?:please\s+)?(?:search(?:\s+for)?|find(?:\s+out)?|tell\s+me\s+about|look\s+up|what\s+is|who\s+is)\s+",
        "",
        raw_query.strip(),
        flags=re.IGNORECASE,
    ).strip(" .?!\"'")
    return cleaned if cleaned else raw_query.strip()


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format raw search dictionary entries into readable markdown blocks, prioritizing authoritative sources."""
    if not results:
        return ""

    # Sort results prioritizing authoritative sources (stable sort preserves order within tiers)
    sorted_results = sorted(
        results,
        key=lambda item: _authority_score(item.get("href") or item.get("url") or item.get("link") or ""),
        reverse=True,
    )

    formatted_entries: List[str] = []
    for idx, item in enumerate(sorted_results, start=1):
        title = item.get("title", "No Title").strip()
        url = item.get("href") or item.get("url") or item.get("link") or "No URL"
        snippet = item.get("body") or item.get("snippet") or item.get("description") or "No description available."
        credibility_label = classify_source_authority(url)

        formatted_entries.append(
            f"[{idx}] {title}\n"
            f"    URL: {url}\n"
            f"    Credibility: {credibility_label}\n"
            f"    Snippet: {snippet.strip()}"
        )
    return "\n\n".join(formatted_entries)


def _search_duckduckgo(query: str, max_results: int) -> str:
    """
    Execute search using DDGS with fast keyword priming, authority sorting, and graceful fallback.
    """
    clean_q = _clean_query(query)

    try:
        ddgs = DDGS(timeout=6)
        results: List[Dict[str, Any]] = []

        # 1. Start with cleaned query if conversational filler was present, else raw query
        primary_q = clean_q if clean_q != query.strip() else query.strip()
        try:
            results = list(ddgs.text(primary_q, max_results=max_results))
        except DDGSException as de:
            if "No results" not in str(de):
                raise
            results = []

        # 2. If empty and primary was cleaned, try raw query as fallback
        if not results and primary_q != query.strip():
            logger.debug("Falling back to raw query: '%s'", query)
            try:
                results = list(ddgs.text(query.strip(), max_results=max_results))
            except DDGSException as de:
                if "No results" not in str(de):
                    raise
                results = []

        # 3. If still empty, try news search as supplementary fallback
        if not results:
            target_q = clean_q or query
            logger.debug("Falling back to news search for: '%s'", target_q)
            try:
                if hasattr(ddgs, "news"):
                    results = list(ddgs.news(target_q, max_results=max_results))
            except Exception:
                results = []

        if not results:
            return f"No search results found for query: '{query}'."

        return _format_search_results(results)

    except TimeoutError as te:
        logger.warning("DuckDuckGo search timeout for '%s': %s", query, str(te))
        return f"Search error (DuckDuckGo): Connection timed out after 6s: {str(te)}"
    except Exception as exc:
        err_msg = str(exc)
        if "No results" in err_msg:
            return f"No search results found for query: '{query}'."
        logger.warning("DuckDuckGo search error for '%s': %s", query, err_msg)
        return f"Search error (DuckDuckGo): Unable to retrieve live web results: {err_msg}"


def _search_tavily(query: str, max_results: int) -> str:
    """Execute search using Tavily API if key is available, with authority sorting."""
    api_key = config.tavily_api_key
    if not api_key:
        logger.info("Tavily API key not found. Falling back to DuckDuckGo.")
        return _search_duckduckgo(query, max_results)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
        raw_results = response.get("results", [])

        if not raw_results:
            return f"No search results found for query: '{query}' via Tavily."

        # Adapt Tavily results schema for _format_search_results
        adapted_results = [
            {
                "title": item.get("title", "No Title"),
                "href": item.get("url", "No URL"),
                "body": item.get("content", "No content available."),
            }
            for item in raw_results
        ]

        return _format_search_results(adapted_results)

    except Exception as exc:
        logger.warning("Tavily search error for '%s': %s. Falling back to DuckDuckGo.", query, str(exc))
        return _search_duckduckgo(query, max_results)


def perform_web_search(
    query: str,
    provider: Optional[str] = None,
    max_results: Optional[int] = None,
) -> str:
    """
    Search the live web using the configured or requested provider.
    Never fabricates search results.
    """
    if not query or not query.strip():
        return "Error: Empty search query provided."

    active_provider = (provider or config.search_provider).lower()
    results_limit = max_results or config.search_max_results

    if active_provider == "tavily":
        return _search_tavily(query.strip(), results_limit)
    return _search_duckduckgo(query.strip(), results_limit)


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the live web in real-time for current information, facts, statistics, news, and external knowledge.
    Use this tool whenever you need up-to-date data, public facts, GDP numbers, population statistics, or recent events.
    Input must be a focused search query string.
    """
    return perform_web_search(query)

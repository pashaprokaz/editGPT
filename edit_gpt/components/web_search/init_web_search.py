from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults


def initialize_web_search(
    web_search: str = "duckduckgo", num_search_results: int = 5, **kwargs
):
    if web_search == "duckduckgo":
        return DuckDuckGoSearchResults(max_results=num_search_results, **kwargs)
    elif web_search == "tavily":
        return TavilySearchResults(max_results=num_search_results, **kwargs)

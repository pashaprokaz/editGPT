from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from edit_gpt.components.chat.agent.tools.edit_file import EditFileTool
from edit_gpt.components.file_storage import BaseFileReader
from edit_gpt.components.rag.base_rag import BaseRAGManager
from edit_gpt.components.web_search.init_web_search import initialize_web_search


def initialize_tools_for_agent(
    web_search_provider: Optional[str] = "tavily",
    edit_files: bool = True,
    web_search_max_results: int = 5,
    rag_manager: Optional[BaseRAGManager] = None,
    chat_model: Optional[BaseChatModel] = None,
    file_reader: Optional[BaseFileReader] = None,
) -> List[BaseTool]:
    tools = []
    if web_search_provider:
        search_tool = initialize_web_search(web_search_provider, web_search_max_results)
        search_tool.name = "web_search"
        search_tool.description = (
            "A search engine optimized for comprehensive, accurate, and trusted results. "
            "Useful for when you need to answer questions about current events. "
            "Input should be a search query."
            "Example:\n"
            "Action: web_search\n"
            'Action Input: {"query": "logging in python"}\n'
        )
        tools.append(search_tool)

    if edit_files:
        if not rag_manager or not chat_model:
            raise ValueError(
                "If you want to use edit_files tool, you need to provide rag_manager, chat_model and file_reader"
            )
        tools.append(
            EditFileTool(
                rag_manager=rag_manager,
                chat_model=chat_model,
                file_reader=file_reader,
            )
        )

    return tools

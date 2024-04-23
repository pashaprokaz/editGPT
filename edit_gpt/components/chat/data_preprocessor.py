from typing import Optional

from langchain_core.tools import BaseTool

from edit_gpt.components.rag.local.rag_local import LocalRAGManager
from edit_gpt.utils.utils import format_docs


class AdditionalDataPreprocessor:
    def __init__(
        self, rag_manager: LocalRAGManager, web_search_tool: Optional[BaseTool] = None
    ):
        self.rag_manager = rag_manager
        self.web_search_tool = web_search_tool

    def prepare_data(self, message: str, options: list[str]) -> dict:
        data = {"rag_context": "", "web_context": ""}
        if "RAG" in options:
            data["rag_context"] = format_docs(
                self.rag_manager.get_filtered_docs(question=message)
            )
        if self.web_search_tool and "Web Search" in options:
            data["web_context"] = self.web_search_tool.invoke(message)
        return data

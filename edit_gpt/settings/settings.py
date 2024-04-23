# TODO add vectorstore settings
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from edit_gpt.settings.settings_loader import load_active_settings


class ChatModel(BaseModel):
    provider: str = Field(
        "fireworks",
        description="You can check the list of available providers in edit_gpt/chat_models/init_chat_model",
    )
    model: str = Field(
        "accounts/fireworks/models/nous-hermes-2-mixtral-8x7b-dpo-fp8",
        description="Model name",
    )
    context_window: Optional[int] = Field(
        None,
        description="The maximum number of context tokens for the model. Required to work with local LLM inferences",
    )
    temperature: float = Field(
        0.1,
        description="The temperature of the model. Increasing the temperature will make the model answer more creatively. A value of 0.1 would be more factual.",
    )

    class Config:
        extra = "allow"


class RagSettings(BaseModel):
    similarity_top_k: int = Field(
        2,
        description="This value controls the number of documents returned by the RAG pipeline",
    )
    filepaths: Optional[List[str]] = Field(
        None,
        description="A list of paths (you can specify folders and files) that will be uploaded to RAG, and which can be edited directly",
    )


class WebSearchSettings(BaseModel):
    provider: Literal["duckduckgo", "tavily"] = Field("tavily")
    k: int = Field(
        2,
        description="This value controls the number of pages returned by the web search tool/pipeline",
    )


class AgentSettings(BaseModel):
    use_web_search: bool
    edit_files: bool
    max_iterations: int = Field(
        5,
        description="The maximum number of iterations for the agent. Includes iterations with an error",
    )
    additional_stop_sequences: List[str] = Field([])


class EmbeddingsSettings(BaseModel):
    model: str


class HistorySettings(BaseModel):
    type: Literal["vector", "simple"] = Field(
        "simple",
        description="The type of the chat history. If simple is set, then the history is stored in-memory in a regular list. In vector mode, k last messages are taken, and n messages are extracted via similarity",
    )

    class Config:
        extra = "allow"


class LangsmithSettings(BaseModel):
    use_langsmith: bool = Field(False)


class Settings(BaseModel):
    chat_model: ChatModel
    rag: RagSettings
    web_search: WebSearchSettings
    agent: AgentSettings
    embeddings: EmbeddingsSettings
    history: HistorySettings
    langsmith: LangsmithSettings


def load_settings() -> Settings:
    data = load_active_settings()
    return Settings(**data)

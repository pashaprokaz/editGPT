from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings.fake import FakeEmbeddings

from edit_gpt.chat_models.init_chat_model import initialize_chat_model
from edit_gpt.components.chat.agent.agent_builder import initialize_agent
from edit_gpt.components.chat.agent.tools.tools_builder import (
    initialize_tools_for_agent,
)
from edit_gpt.components.chat.chat_manager import ChatManager
from edit_gpt.components.chat.chat_prompts import qa_prompt
from edit_gpt.components.chat.data_preprocessor import AdditionalDataPreprocessor
from edit_gpt.components.diff_storage import DiffReader, DiffStorage
from edit_gpt.components.history.init_history import initialize_history
from edit_gpt.components.ingest_service import IngestService
from edit_gpt.components.langsmith_client import setup_langsmith_client
from edit_gpt.components.rag.local.rag_local import LocalRAGManager
from edit_gpt.components.web_search.init_web_search import initialize_web_search
from edit_gpt.settings.settings import load_settings
from edit_gpt.ui.ui import EditGptUi


def launch_app():
    load_dotenv()

    settings = load_settings()

    if settings.langsmith.use_langsmith:
        setup_langsmith_client()

    chat_model = initialize_chat_model(**settings.chat_model.model_dump())
    if settings.embeddings.model == "fake":
        embeddings = FakeEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings.model)
    rag_manager = LocalRAGManager(embeddings=embeddings, chat_model=chat_model)
    ingest_service = IngestService(rag_manager)
    diff_storage = DiffStorage()
    history = initialize_history(
        history_type=settings.history.type,
        embeddings=embeddings,
        n_retrieved_messages=settings.history.n,
    )
    tools = initialize_tools_for_agent(
        rag_manager=rag_manager,
        chat_model=chat_model,
        web_search_max_results=settings.web_search.k,
        web_search_provider=(
            settings.web_search.provider if settings.agent.use_web_search else None
        ),
        edit_files=settings.agent.edit_files,
        file_reader=DiffReader(diff_storage),
    )
    agent = initialize_agent(
        chat_model=chat_model,
        tools=tools,
        agent_max_iterations=settings.agent.max_iterations,
        additional_stop_sequences=settings.agent.additional_stop_sequences,
    )
    web_search_tool = initialize_web_search(
        settings.web_search.provider, settings.web_search.k
    )
    chat_manager = ChatManager(
        chat_model=chat_model,
        qa_prompt=qa_prompt,
        history=history,
        agent=agent,
    )
    additional_data_manager = AdditionalDataPreprocessor(
        rag_manager=rag_manager, web_search_tool=web_search_tool
    )
    ui = EditGptUi(
        ingest_service=ingest_service,
        chat_manager=chat_manager,
        filenames=settings.rag.filepaths,
        additional_data_manager=additional_data_manager,
        diff_storage=diff_storage,
    )
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)


if __name__ == "__main__":
    launch_app()

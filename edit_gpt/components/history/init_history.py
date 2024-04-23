import uuid
from typing import Literal

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

from edit_gpt.components.history.vector_based_history import VectorBasedChatHistory


def initialize_history(
    history_type: Literal["vector", "simple"] = "simple",
    vectorstore=FAISS,
    embeddings=HuggingFaceEmbeddings(),
    n_retrieved_messages: int = 3,
):
    chat_message_history = ChatMessageHistory()
    if history_type == "simple":
        return chat_message_history

    # init empty db and retriever
    fake_id = str(uuid.uuid4())
    vector_history_db = vectorstore.from_documents(
        [Document(page_content="")], embeddings, ids=[fake_id]
    )
    vector_history_db.delete([fake_id])
    vector_history_db_retriever = vector_history_db.as_retriever(
        search_kwargs={"k": n_retrieved_messages}
    )

    vector_history = VectorBasedChatHistory(
        vector_history_db_retriever, chat_message_history
    )
    return vector_history

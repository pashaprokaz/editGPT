import re
import uuid
from typing import List, Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever


class VectorBasedChatHistory(BaseChatMessageHistory):
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        chat_message_history: BaseChatMessageHistory,
        k_last_messages: Optional[int] = 2,
    ):
        self.retriever = retriever
        self.chat_message_history = chat_message_history
        self.k_last_messages = k_last_messages
        self.docs_in_retriever_ids = []

    @property
    def messages(self):
        return self.chat_message_history.messages

    def add_message(self, message: BaseMessage) -> None:
        self.chat_message_history.add_message(message)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(
            [f"<mtype>{message.type}<mtype> {message.content}"]
        )
        ids = [str(uuid.uuid4()) for _ in docs]
        for i, doc in enumerate(docs):
            doc.metadata["id"] = ids[i]
        self.docs_in_retriever_ids.extend(ids)
        self.retriever.add_documents(docs, ids=ids)

    @staticmethod
    def db_doc_to_message(db_doc):
        match = re.search(r"<mtype>(.*?)<mtype>", db_doc.page_content)
        if match:
            type_ = match.group(1)
        else:
            type_ = None
        content = re.sub(r"<mtype>.*?<mtype>", "", db_doc.page_content).strip()
        if type_ == "human":
            return HumanMessage(content=content)
        elif type_ == "ai":
            return AIMessage(content=content)
        else:
            return SystemMessage(content=content)

    def clear(self) -> None:
        self.chat_message_history.clear()
        if self.docs_in_retriever_ids:
            self.retriever.vectorstore.delete(self.docs_in_retriever_ids)
            self.docs_in_retriever_ids = []

    def retrieve(self, question: str) -> List[BaseMessage]:
        """
        Retrieves a list of messages related to the given question.

        This method first retrieves the last few messages from the chat history,
        then invokes the retriever to get a list of relevant messages from the database.
        It filters out any messages that are already in the last few messages,
        and then returns the combined list of retrieved messages and last few messages.

        Args:
            question (str): The question to retrieve related messages for.

        Returns:
            List[BaseMessage]: A list of messages related to the question.
        """
        last_messages = []
        if self.k_last_messages:
            last_messages = self.chat_message_history.messages[-self.k_last_messages :]
            for message in last_messages:
                question += "\n" + message.content

        last_messages_content = [message.content for message in last_messages]

        retrieved_history = self.retriever.invoke(question)

        retrieved_history_messages = []
        for db_doc in retrieved_history:
            message_content = self.db_doc_to_message(db_doc).content
            if message_content not in last_messages_content:
                message = self.db_doc_to_message(db_doc)
                retrieved_history_messages.append(message)

        return retrieved_history_messages + last_messages

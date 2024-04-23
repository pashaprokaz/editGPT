from typing import Any, Dict, Generator, List, Optional, Union

from langchain.agents import AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import BaseChatPromptTemplate

from edit_gpt.components.history.vector_based_history import VectorBasedChatHistory


class ChatManager:
    def __init__(
        self,
        chat_model: BaseChatModel,
        qa_prompt: BaseChatPromptTemplate,
        history: Optional[Union[BaseChatMessageHistory, VectorBasedChatHistory]] = None,
        agent: Optional[AgentExecutor] = None,
    ):
        self.chat_model = chat_model
        self.history = history
        self.qa_prompt = qa_prompt
        self.agent = agent

    def get_answer(
        self, question: str, history: Optional[List[BaseMessage]] = None, **kwargs
    ):
        qa_chain = self.qa_prompt | self.chat_model

        return qa_chain.stream(
            {
                "question": question,
                "history": history,
                **kwargs,
            }
        )

    def chat_gen(self, text: str, **kwargs) -> Generator[str, None, None]:
        accumulated_text = ""

        history = []
        if self.history:
            if isinstance(self.history, VectorBasedChatHistory):
                history = self.history.retrieve(text)
            elif isinstance(self.history, BaseChatMessageHistory):
                history = self.history.messages

        for chunk in self.get_answer(text, history=history, **kwargs):
            chunk_content = chunk.content
            accumulated_text += chunk_content

            yield accumulated_text

        if self.history:
            self.history.add_message(HumanMessage(content=text))
            self.history.add_message(AIMessage(content=accumulated_text))

    def update_history(self, messages):
        if self.history:
            self.history.clear()
            for message in messages:
                self.history.add_message(message)

    def agent_gen(self, text: str, **kwargs) -> Dict[str, Any]:
        history_messages = []
        if self.history:
            if isinstance(self.history, VectorBasedChatHistory):
                history_messages = self.history.retrieve(text)
            elif isinstance(self.history, BaseChatMessageHistory):
                history_messages = self.history.messages

        result = self.agent.invoke(
            {
                "input": text,
                "history": history_messages,
                "get_final_answer_message": "",
                **kwargs,
            }
        )

        if self.history:
            self.history.add_message(HumanMessage(content=text))
            self.history.add_message(AIMessage(content=result["output"]))

        return result

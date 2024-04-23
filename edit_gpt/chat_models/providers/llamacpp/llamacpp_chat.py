from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class LlamaCppChat(BaseChatModel):
    model: Any
    temperature: float
    top_p: float

    @staticmethod
    def format_messages_to_llava_cpp(
        messages: List[BaseMessage],
    ) -> List[Dict[str, str]]:
        formatted_messages = []
        for message in messages:
            role = message.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            formatted_messages.append({"role": role, "content": message.content})
        return formatted_messages

    @classmethod
    def from_model_path(
        cls,
        model_path=None,
        temperature=0.4,
        top_p=0.7,
        n_gpu_layers=-1,
        chat_format="llama-2",
        n_ctx=2048,
        **kwargs,
    ):
        from llamacpp import Llama

        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            n_ctx=n_ctx,
            **kwargs,
        )

        return cls(model=model, temperature=temperature, top_p=top_p)

    @property
    def _llm_type(self) -> str:
        return "llamacpp-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = self.format_messages_to_llava_cpp(messages)
        raw_output = self.model.create_chat_completion(
            messages=messages, temperature=self.temperature, top_p=self.top_p, stop=stop
        )
        output = raw_output["choices"][0]["message"]["content"]
        chat_generation = ChatGeneration(message=AIMessage(content=output))
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        messages = self.format_messages_to_llava_cpp(messages)
        output = self.model.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
            stop=stop,
        )

        for new_text in output:
            if "content" in new_text["choices"][0]["delta"]:
                text_chunk = new_text["choices"][0]["delta"]["content"]
                yield ChatGenerationChunk(message=AIMessageChunk(content=text_chunk))

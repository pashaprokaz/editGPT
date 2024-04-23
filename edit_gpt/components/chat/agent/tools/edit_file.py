from operator import attrgetter
from typing import Any, List, Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from edit_gpt.components.file_storage import BaseFileReader
from edit_gpt.utils.loaders import normalize_to_straight_slash

SUCCESSFUL_EDIT_MESSAGE = "File was successfully changed. New file content: \n"
FAILED_EDIT_MESSAGE = "File was not changed. User refused to edit the file"


def build_successful_observation(text):
    return f"{SUCCESSFUL_EDIT_MESSAGE}{text}"


def get_text_from_observation(observation):
    return observation.split(SUCCESSFUL_EDIT_MESSAGE)[1]


class EditFileInput(BaseModel):
    query: str = Field(..., description="User request to edit the file")
    path: str = Field(..., description="Path to the file that should be edited")


class EditFileTool(BaseTool):
    name = "edit_file_with_llm"
    description = (
        "Call another language model that will change the contents of the file based on detailed "
        "description and path to the file."
        "\nExample:"
        "\nAction: edit_file_with_llm"
        '\nAction Input: {"query": '
        '"(a detailed description of how you would change the file))", "path": "(the full path to the file))"}\n'
    )
    args_schema: Type[BaseModel] = EditFileInput
    rag_manager: Any
    chat_model: BaseChatModel
    file_reader: BaseFileReader

    def _run(self, query: str, path: str) -> str:
        docs = self.get_docs_from_filename(path.strip())
        if len(docs) == 0:
            return "Provided path does not exist."
        result = self.change_file_content(
            docs[0].metadata["source"], query, self.chat_model
        )
        if result:
            return "File was successfully changed. New file content: \n" + result
        else:
            return "File was not changed. User refused to edit the file"

    def get_docs_from_filename(self, filename: str) -> List[Document]:
        normalized_filename = normalize_to_straight_slash(filename)
        all_docs = self.rag_manager.get_all_docs()
        result_docs = []
        for doc in all_docs:
            if doc.metadata["source"] == normalized_filename:
                result_docs.append(doc)

        return result_docs

    def change_file_content(
        self, file_name: str, user_request: str, chat_model: BaseChatModel
    ) -> str:
        get_corrections_system_prompt = (
            "You will be presented with the source code and the user's request. "
            "In response, send the modified code in full, without explanations and another comments."
        )
        file_content = self.file_reader.read_from_filename(file_name)

        get_corrections_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", get_corrections_system_prompt),
                HumanMessage(content=file_content),
                ("human", "{question}"),
            ]
        )

        get_corrections_chain = (
            get_corrections_prompt | chat_model | attrgetter("content")
        )

        content = get_corrections_chain.invoke({"question": user_request})

        return content

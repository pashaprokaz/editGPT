from langchain_core.prompts import ChatPromptTemplate

from edit_gpt.components.chat.chat_prompts import RAG_SYSTEM_PROMPT

FILTER_SYSTEM_PROMPT = (
    "Below will be presented the user's request and information from the user's"
    " documents that may be useful obtained through RAG. You "
    "must determine which of the specified documents presented in the RAG can be useful for "
    "solving the user's task. Write ONLY the indexes of the files without additional "
    "explanations"
)

FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", FILTER_SYSTEM_PROMPT),
        ("human", "USER'S REQUEST: {question}"),
        ("human", RAG_SYSTEM_PROMPT),
    ]
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

QA_SYSTEM_PROMPT = (
    "Below will be presented: the user's request, "
    "information from the user's documents that may be useful obtained through RAG, and, "
    "if possible, information obtained from other sources."
    "Constructively answer the user's question. If necessary, use the information provided."
)
RAG_SYSTEM_PROMPT = "\nRAG TEXT START:\n\n{rag_context}\n\nRAG TEXT END\n"
WEB_SEARCH_SYSTEM_PROMPT = (
    "\nWEB SEARCH TEXT START:\n\n{web_context}\n\nWEB SEARCH TEXT END\n"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "USER'S REQUEST: {question}"),
        ("human", RAG_SYSTEM_PROMPT),
        ("human", WEB_SEARCH_SYSTEM_PROMPT),
    ]
)

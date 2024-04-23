def openai_provider(model: str, **kwargs):
    from langchain_openai import ChatOpenAI  # type: ignore

    return ChatOpenAI(model=model, **kwargs)

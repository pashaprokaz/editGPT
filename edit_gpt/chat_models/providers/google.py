def google_provider(model: str, **kwargs):
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

    return ChatGoogleGenerativeAI(model=model, **kwargs)

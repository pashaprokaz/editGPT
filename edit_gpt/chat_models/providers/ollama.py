def ollama_provider(
    model: str,
    temperature: float = 0.45,
    top_p: float = 0,
    **kwargs,
):
    from langchain_community.chat_models import ChatOllama

    return ChatOllama(model=model, temperature=temperature, top_p=top_p, **kwargs)

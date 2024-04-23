def groq_provider(model: str, temperature: float = 0.7, **_):
    from langchain_groq import ChatGroq

    return ChatGroq(model_name=model, temperature=temperature)

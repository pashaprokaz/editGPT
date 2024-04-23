def fireworks_provider(
    model: str,
    temperature: float = 0.45,
    top_p: float = 0.7,
    max_tokens: int = 2048,
    **_,
):
    from langchain_fireworks import ChatFireworks

    return ChatFireworks(
        model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

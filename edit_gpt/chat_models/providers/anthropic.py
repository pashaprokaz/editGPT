from langchain_community.chat_models.anthropic import ChatAnthropic


def anthropic_provider(model: str, **kwargs):
    return ChatAnthropic(model=model, **kwargs)

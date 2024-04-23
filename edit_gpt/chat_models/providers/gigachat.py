from langchain_community.chat_models import GigaChat


def gigachat_provider(
    model: str = "GigaChat-Pro", verify_ssl_certs: bool = False, **kwargs
):
    return GigaChat(verify_ssl_certs=verify_ssl_certs, model=model, **kwargs)

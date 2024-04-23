def fake_provider(**kwargs):
    from langchain_core.language_models.fake_chat_models import FakeChatModel

    return FakeChatModel(**kwargs)

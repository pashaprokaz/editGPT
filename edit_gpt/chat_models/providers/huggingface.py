from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_hub import HuggingFaceHub


def huggingface_provider(model: str, task="text-generation", **kwargs):
    llm = HuggingFaceHub(
        repo_id=model,
        task=task,
        model_kwargs={
            **kwargs,
        },
    )
    return ChatHuggingFace(llm=llm)

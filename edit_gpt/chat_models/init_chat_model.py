from edit_gpt.chat_models.providers.anthropic import anthropic_provider
from edit_gpt.chat_models.providers.fake import fake_provider
from edit_gpt.chat_models.providers.fireworks import fireworks_provider
from edit_gpt.chat_models.providers.gigachat import gigachat_provider
from edit_gpt.chat_models.providers.google import google_provider
from edit_gpt.chat_models.providers.groq import groq_provider
from edit_gpt.chat_models.providers.huggingface import huggingface_provider
from edit_gpt.chat_models.providers.llamacpp.llamacpp import llamacppchat_provider
from edit_gpt.chat_models.providers.ollama import ollama_provider
from edit_gpt.chat_models.providers.openai import openai_provider

PROVIDERS_MAP = {
    "llamacpp": llamacppchat_provider,
    "gigachat": gigachat_provider,
    "fireworks": fireworks_provider,
    "anthropic": anthropic_provider,
    "google": google_provider,
    "huggingface": huggingface_provider,
    "openai": openai_provider,
    "ollama": ollama_provider,
    "groq": groq_provider,
    "fake": fake_provider,
}


def initialize_chat_model(provider: str, model: str, **kwargs):
    provider_function = PROVIDERS_MAP[provider]
    chat_model = provider_function(model=model, **kwargs)
    return chat_model

from edit_gpt.chat_models.providers.llamacpp.llamacpp_chat import LlamaCppChat


def llamacppchat_provider(
    model: str,
    n_gpu_layers: int = -1,
    chat_format: str = "llama-2",
    n_ctx: int = 2048,
    **kwargs,
):
    return LlamaCppChat.from_model_path(
        model_path=model,
        n_gpu_layers=n_gpu_layers,
        chat_format=chat_format,
        n_ctx=n_ctx,
        **kwargs,
    )

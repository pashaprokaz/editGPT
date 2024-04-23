from edit_gpt.settings.settings import load_settings


def test_settings_are_loaded_and_merged() -> None:
    assert load_settings().chat_model.provider == "fake"

import functools
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic.v1.utils import deep_update, unique_list

from edit_gpt.constants import PROJECT_ROOT_PATH
from edit_gpt.settings.yaml_loader import load_yaml_with_envvars

logger = logging.getLogger(__name__)


def merge_settings(settings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return functools.reduce(deep_update, settings, {})


def load_settings_from_profile(profile: str) -> dict[str, Any]:
    _settings_folder = os.environ.get("EGPT_SETTINGS_FOLDER", PROJECT_ROOT_PATH)
    if profile == "default":
        profile_file_name = "settings.yaml"
    else:
        profile_file_name = f"settings-{profile}.yaml"

    path = Path(_settings_folder) / profile_file_name
    with Path(path).open("r") as f:
        config = load_yaml_with_envvars(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file has no top-level mapping: {path}")
    return config


def load_active_settings() -> dict[str, Any]:
    """Load active profiles and merge them."""
    # if running in unittest, use the test profile
    _test_profile = ["test"] if "tests" in sys.modules else []
    egpt_profiles = os.environ.get("EGPT_PROFILES", "")
    egpt_profiles = egpt_profiles.split(",") if egpt_profiles is not None else []

    active_profiles: list[str] = unique_list(
        ["default"]
        + [item.strip() for item in egpt_profiles if item.strip()]
        + _test_profile
    )

    logger.info("Starting application with profiles=%s", active_profiles)
    loaded_profiles = [
        load_settings_from_profile(profile) for profile in active_profiles
    ]
    merged: dict[str, Any] = merge_settings(loaded_profiles)
    return merged

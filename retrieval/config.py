"""Load path settings from config.yaml at the repo root."""

from __future__ import annotations

from pathlib import Path

import yaml

_PATH_KEYS = ("specter_model", "chroma", "collection", "cross_encoder")


def repo_root() -> Path:
    """Return the directory that contains pyproject.toml."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not find pyproject.toml above the config module.")


def load_config(path: Path | None = None) -> dict:
    """Read config.yaml.

    Args:
        path: Config file to read. Defaults to config.yaml in the repo root.

    Returns:
        The mapping stored in that file.
    """
    config_path = path or (repo_root() / "config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must be a mapping: {config_path}")
    return loaded


def configured_paths(path: Path | None = None) -> dict[str, str]:
    """Return the paths section, with folder paths resolved to absolute paths.

    Args:
        path: Config file to read. Defaults to config.yaml in the repo root.

    Returns:
        specter_model, chroma, collection, and cross_encoder.
    """
    config_path = path or (repo_root() / "config.yaml")
    paths = load_config(config_path).get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Config file needs a paths section: {config_path}")

    resolved: dict[str, str] = {}
    for key in _PATH_KEYS:
        if key not in paths:
            raise ValueError(f"Config paths.{key} is missing: {config_path}")
        value = str(paths[key])
        if key == "collection":
            resolved[key] = value
            continue
        folder = Path(value)
        if not folder.is_absolute():
            folder = config_path.parent / folder
        resolved[key] = str(folder.resolve())
    return resolved

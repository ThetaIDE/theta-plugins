"""Rulesets: entries of the ``rules`` workspace resource this package declares.

An entry is ``<environment>/<name>`` under ``src/usr/rules`` or this package's
``resources/rules``. A selection is that full entry, or a bare name resolved
against the experiment environment's ``symbolic_name`` (else its last path
segment). A ruleset may supply its own ``valuation.py`` through ``ruleset.yaml``;
otherwise the environment's declared resource provides it.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.app.api.packages import environment_resource, resource_entries
from src.app.api.workspace import WorkspacePaths

RESOURCE = "rules"


@dataclass(frozen=True)
class Ruleset:
    environment: str
    name: str
    directory: Path

    @property
    def metadata(self):
        path = self.directory / "ruleset.yaml"
        data = yaml.safe_load(path.read_text()) if path.is_file() else {}
        if data is not None and not isinstance(data, dict):
            raise ValueError(f"Ruleset metadata must be a mapping: {path}")
        if data and data.get("format_version", 1) != 1:
            raise ValueError(f"Unsupported ruleset format: {path}")
        return data or {}

    @property
    def valuation_path(self):
        source = self.metadata.get("valuation")
        if source:
            path = (self.directory / str(source)).resolve()
            if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
                raise ValueError(f"Ruleset valuation must be a file inside {self.directory}")
            return path
        return environment_resource(self.environment, "valuation.py")


def _segment(name):
    if not name or "\\" in name or Path(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"Invalid ruleset selection: {name!r}")
    return name


def symbolic_environment(environment):
    """The symbolic environment of an env selection: its symbolic_name, else its last path segment."""
    if isinstance(environment, Mapping):
        if environment.get("symbolic_name"):
            return _segment(str(environment["symbolic_name"]))
        environment = environment.get("name")
    if not environment:
        return None
    name = _segment(str(environment))
    paths = WorkspacePaths.for_process()
    for suffix in (".yaml", ".yml"):
        profile = paths.resolve_config("env/" + name + suffix)
        if profile.is_file():
            data = yaml.safe_load(profile.read_text()) or {}
            if data.get("symbolic_name"):
                return _segment(str(data["symbolic_name"]))
    return Path(name).name


def resolve_ruleset(environment, selection) -> Ruleset:
    """The ruleset a model's ``rules`` value selects for the given environment selection."""
    if not isinstance(selection, str) or not selection:
        raise ValueError("Select one ruleset")
    parts = _segment(selection).split("/")
    if len(parts) == 2:
        # A full entry names its environment; compatibility with the simulator is the author's choice.
        env, name = parts
    elif len(parts) == 1:
        env, name = symbolic_environment(environment), parts[0]
        if not env:
            raise ValueError("Select an environment or a full environment/ruleset entry")
    else:
        raise ValueError("Ruleset selections are a name or environment/name")
    directory = resource_entries(RESOURCE).get(f"{env}/{name}")
    if directory is None:
        raise ValueError(f"No ruleset {name!r} for environment {env!r}")
    return Ruleset(env, name, directory)

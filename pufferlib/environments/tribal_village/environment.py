"""
Tribal Village Environment PufferLib Integration.

Simple integration that imports the tribal-village environment directly.
The tribal-village package should be installed via pip.
"""

import functools
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pufferlib


def _import_tribal_village_env():
    """Import helper that falls back to the local tribal-village checkout."""

    try:
        from tribal_village_env.environment import TribalVillageEnv  # type: ignore
        return TribalVillageEnv
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        fallback_dir = repo_root.parent / 'tribal-village'
        if fallback_dir.exists() and str(fallback_dir) not in sys.path:
            sys.path.insert(0, str(fallback_dir))
        try:
            from tribal_village_env.environment import TribalVillageEnv  # type: ignore
            return TribalVillageEnv
        except ImportError as exc:
            raise ImportError(
                "Failed to import tribal-village environment. Install the package with\n"
                "  pip install pufferlib[tribal-village] --no-build-isolation\n"
                "or keep a checkout at ../tribal-village containing tribal_village_env/."
            ) from exc


def env_creator(name='tribal_village'):
    return functools.partial(make, name=name)


def make(name='tribal_village', config=None, buf=None, **kwargs):
    """Create a tribal village PufferLib environment instance."""

    TribalVillageEnv = _import_tribal_village_env()

    # Merge config with kwargs
    if config is None:
        config = {}
    config = {**config, **kwargs}

    # Create the environment
    env = TribalVillageEnv(config=config, buf=buf)
    return env


# Configuration should come from the tribal-village repository's canonical config
# No default overrides here - use tribal-village's own configuration

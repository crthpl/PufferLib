"""
Tribal Village Environment PufferLib Integration.

Simple integration that imports the tribal-village environment directly.
The tribal-village package should be installed via pip.
"""

import functools
from typing import Any, Dict, Optional

import pufferlib


def env_creator(name='tribal_village'):
    return functools.partial(make, name=name)


def make(name='tribal_village', config=None, buf=None, **kwargs):
    """Create a tribal village PufferLib environment instance."""

    try:
        # Import the installed tribal village environment
        from tribal_village_env.environment import TribalVillageEnv

        # Merge config with kwargs
        if config is None:
            config = {}
        config = {**config, **kwargs}

        # Create the environment
        env = TribalVillageEnv(config=config, buf=buf)
        return env

    except ImportError as e:
        raise ImportError(
            f"Failed to import tribal-village environment: {e}\n\n"
            "Install the tribal-village environment with:\n"
            "  pip install pufferlib[tribal-village] --no-build-isolation"
        ) from e


# Configuration should come from the tribal-village repository's canonical config
# No default overrides here - use tribal-village's own configuration
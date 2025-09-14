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


# Default configuration for tribal village environment
TRIBAL_VILLAGE_CONFIG = {
    'max_steps': 512,
    'ore_per_battery': 3,
    'batteries_per_heart': 2,
    'enable_combat': True,
    'clippy_spawn_rate': 0.1,
    'clippy_damage': 10,
    'heart_reward': 1.0,
    'battery_reward': 0.5,
    'ore_reward': 0.1,
    'survival_penalty': -0.01,
    'death_penalty': -1.0,
}
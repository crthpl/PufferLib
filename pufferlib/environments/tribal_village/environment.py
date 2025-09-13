"""
Tribal Village Environment PufferLib Integration.

This provides PufferLib compatibility for the Tribal Village environment,
a multi-agent reinforcement learning environment built with Nim.
"""

import functools
from typing import Any, Dict, Optional

import pufferlib


def env_creator(name='tribal_village'):
    return functools.partial(make, name=name)


def make(name='tribal_village', config=None, buf=None, **kwargs):
    """Create a tribal village PufferLib environment instance."""

    try:
        # Import from the external tribal-village-env package
        from tribal_village_env import TribalVillageEnv

        # Merge config with kwargs
        if config is None:
            config = {}
        config = {**config, **kwargs}

        # Create the environment - it already implements PufferEnv interface
        env = TribalVillageEnv(config=config, buf=buf)

        return env

    except ImportError as e:
        raise ImportError(
            f"Failed to import tribal-village-env: {e}\\n\\n"
            "This environment requires the tribal-village-env package. "
            "Install with: pip install tribal-village-env\\n"
            "Or from source: pip install git+https://github.com/Metta-AI/tribal-village.git"
        ) from e


# Default configuration for tribal environment
TRIBAL_CONFIG = {
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
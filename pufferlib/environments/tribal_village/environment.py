"""
Tribal Village Environment PufferLib Integration.

This provides PufferLib compatibility for the Tribal Village environment,
a multi-agent reinforcement learning environment built with Nim.

The environment requires the tribal-village repository to be cloned and built.
"""

import functools
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pufferlib


def env_creator(name='tribal_village'):
    return functools.partial(make, name=name)


def make(name='tribal_village', config=None, buf=None, **kwargs):
    """Create a tribal village PufferLib environment instance."""

    try:
        # This assumes tribal-village repo is cloned and accessible
        # Users should: git clone https://github.com/Metta-AI/tribal-village.git

        # Try to find the tribal-village repository
        possible_paths = [
            Path("tribal-village"),
            Path("../tribal-village"),
            Path.home() / "tribal-village",
            Path.cwd() / "tribal-village"
        ]

        tribal_village_path = None
        for path in possible_paths:
            if path.exists() and (path / "build_lib.sh").exists():
                tribal_village_path = path
                break

        if tribal_village_path is None:
            raise ImportError(
                "Tribal Village repository not found. Please clone it:\n"
                "git clone https://github.com/Metta-AI/tribal-village.git\n"
                "Then run: cd tribal-village && ./build_lib.sh"
            )

        # Add the tribal-village path to Python path
        sys.path.insert(0, str(tribal_village_path))

        # Import the ctypes-based environment
        from tribal_village_env.environment import TribalVillageEnv

        # Merge config with kwargs
        if config is None:
            config = {}
        config = {**config, **kwargs}

        # Create the environment - it already implements PufferEnv interface
        env = TribalVillageEnv(config=config, buf=buf)

        return env

    except ImportError as e:
        raise ImportError(
            f"Failed to import tribal-village environment: {e}\n\n"
            "This environment requires the tribal-village repository. "
            "Clone and build it with:\n"
            "  git clone https://github.com/Metta-AI/tribal-village.git\n"
            "  cd tribal-village && ./build_lib.sh"
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
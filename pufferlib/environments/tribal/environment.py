"""
Tribal Environment PufferLib Integration.

This provides PufferLib compatibility for the Tribal environment using
genny-generated bindings from the Metta AI project.
"""

import functools
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pufferlib
import pufferlib.emulation


def env_creator(name='tribal'):
    return functools.partial(make, name=name)


def make(name='tribal', config=None, render_mode='rgb_array', buf=None, seed=0):
    """Create a tribal PufferLib environment instance."""
    
    # Try to import the TribalPufferEnv from the metta project
    try:
        # This assumes the metta project is installed or accessible
        from metta.sim.tribal_puffer import TribalPufferEnv
        
        # Create the environment with the provided config
        env = TribalPufferEnv(config=config, render_mode=render_mode, buf=buf)
        
        # Wrap with episode stats for PufferLib compatibility
        env = pufferlib.EpisodeStats(env)
        
        return env
        
    except ImportError as e:
        raise ImportError(
            f"Failed to import TribalPufferEnv: {e}\\n\\n"
            "This environment requires the Metta AI project with tribal bindings. "
            "Please ensure the metta package is installed and tribal bindings are built."
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
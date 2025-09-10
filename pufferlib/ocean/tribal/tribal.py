"""
Tribal Environment Ocean Integration.

This provides the 'puffer_tribal' environment for the Ocean registry,
allowing tribal to be used via PufferLib's CLI system.
"""

import pufferlib
import pufferlib.emulation


class Tribal:
    """Factory class for Tribal environment via Ocean registry."""
    
    def __init__(self, buf=None, **config):
        """
        Create a tribal environment through the Ocean registry system.
        
        Args:
            buf: PufferLib buffer for vectorization
            **config: Configuration parameters for the tribal environment
        """
        try:
            # Import the TribalPufferEnv
            from metta.sim.tribal_puffer import TribalPufferEnv
            
            # Create the environment
            env = TribalPufferEnv(config=config, buf=buf)
            
            # Wrap with episode stats for Ocean compatibility
            env = pufferlib.EpisodeStats(env)
            
            # Store the environment
            self.env = env
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import TribalPufferEnv: {e}\\n\\n"
                "This environment requires the Metta AI project with tribal bindings. "
                "Please ensure the metta package is installed and tribal bindings are built."
            ) from e
    
    def __getattr__(self, name):
        """Delegate all attribute access to the underlying environment."""
        return getattr(self.env, name)
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


# For direct module import compatibility
def make_tribal(buf=None, **config):
    """Direct factory function for tribal environment."""
    return Tribal(buf=buf, **config)
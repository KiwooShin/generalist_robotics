"""Cross-embodiment locomotion: walking a policy through morphology space by continuation."""

from generalist_robotics.runtime.gpu import configure_jax_memory

__version__ = "0.1.0"

# Applied at import so no submodule can initialise JAX with the 75% default
# preallocation, which reserves ~116 GiB of this machine's 121 GiB.
configure_jax_memory()

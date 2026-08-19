"""GPU memory guards for the DGX Spark's unified CPU/GPU memory."""

import contextlib
import fcntl
import os
import pathlib
import time
from collections.abc import Iterator

DEFAULT_MEM_FRACTION = 0.25
LOCK_PATH = pathlib.Path(os.environ.get("GENROBO_GPU_LOCK", "/tmp/genrobo-gpu.lock"))


class GpuBusyError(RuntimeError):
    """Raised when another process already holds the GPU lock."""


def configure_jax_memory(mem_fraction: float = DEFAULT_MEM_FRACTION) -> dict[str, str]:
    """Cap JAX device memory before the backend initialises, and report the settings.

    On the DGX Spark, "GPU" memory is system RAM. JAX preallocates 75% of it by
    default, which reserves ~116 GiB of a 121 GiB machine in a single process;
    two such processes exhaust the machine and wedge in the NVIDIA driver, where
    the kernel OOM killer cannot reap them. Existing environment values win, so a
    caller can still opt into larger limits deliberately.
    """
    if not 0.0 < mem_fraction <= 1.0:
        raise ValueError(f"mem_fraction must be in (0, 1], got {mem_fraction}")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
    }


def jax_already_imported() -> bool:
    """Return True when jax is imported, after which memory settings no longer apply."""
    import sys

    return "jax" in sys.modules


@contextlib.contextmanager
def gpu_lock(timeout_seconds: float = 0.0, poll_seconds: float = 2.0) -> Iterator[pathlib.Path]:
    """Hold an exclusive advisory lock so only one process runs GPU work at a time.

    Concurrent JAX processes are what took the machine down; this makes the
    serialisation explicit instead of relying on discipline. With the default
    timeout of zero the call fails immediately rather than queueing.
    """
    LOCK_PATH.touch(exist_ok=True)
    handle = LOCK_PATH.open("r+")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise GpuBusyError(
                    f"another process holds {LOCK_PATH}; GPU work must be serialised "
                    "on unified memory. Wait for it or raise timeout_seconds."
                ) from None
            time.sleep(poll_seconds)
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield LOCK_PATH
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

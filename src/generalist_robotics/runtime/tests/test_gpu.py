"""Unit tests for the GPU memory and serialisation guards."""

import multiprocessing
import os
import pathlib
import tempfile
import unittest

from generalist_robotics.runtime import gpu


class ConfigureJaxMemoryTest(unittest.TestCase):
    """Checks the JAX memory caps are applied without overriding explicit settings."""

    def setUp(self):
        self.saved = {
            key: os.environ.get(key)
            for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")
        }
        for key in self.saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_disables_preallocation_and_caps_the_fraction(self):
        settings = gpu.configure_jax_memory(0.25)
        self.assertEqual(settings["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertEqual(settings["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.25")
        self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")

    def test_existing_environment_values_win(self):
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        settings = gpu.configure_jax_memory(0.25)
        self.assertEqual(settings["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.9")

    def test_rejects_out_of_range_fractions(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                gpu.configure_jax_memory(bad)


def hold_lock(path, started, release):
    """Child-process helper: take the lock, signal, wait, then drop it."""
    os.environ["GENROBO_GPU_LOCK"] = str(path)
    import importlib

    from generalist_robotics.runtime import gpu as child_gpu

    importlib.reload(child_gpu)
    with child_gpu.gpu_lock():
        started.set()
        release.wait(timeout=30)


class GpuLockTest(unittest.TestCase):
    """Checks the advisory lock actually excludes a second process."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "gpu.lock"
        os.environ["GENROBO_GPU_LOCK"] = str(self.path)
        import importlib

        importlib.reload(gpu)

    def tearDown(self):
        os.environ.pop("GENROBO_GPU_LOCK", None)
        self.tmp.cleanup()

    def test_lock_is_reentrant_after_release(self):
        with gpu.gpu_lock():
            pass
        with gpu.gpu_lock():
            pass

    def test_records_the_holding_pid(self):
        with gpu.gpu_lock() as path:
            self.assertEqual(path.read_text().strip(), str(os.getpid()))

    def test_second_process_is_refused(self):
        ctx = multiprocessing.get_context("spawn")
        started, release = ctx.Event(), ctx.Event()
        child = ctx.Process(target=hold_lock, args=(self.path, started, release))
        child.start()
        try:
            self.assertTrue(started.wait(timeout=60), "child never acquired the lock")
            with self.assertRaises(gpu.GpuBusyError):
                with gpu.gpu_lock():
                    pass
        finally:
            release.set()
            child.join(timeout=30)


if __name__ == "__main__":
    unittest.main()

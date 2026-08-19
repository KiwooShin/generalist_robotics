"""Unit tests for cross-embodiment environment construction."""

import unittest

import numpy as np

from generalist_robotics.manipulation import embodiments
from generalist_robotics.manipulation.env_factory import action_dim, arm_dof, make_env
from generalist_robotics.manipulation import robosuite_supported


@unittest.skipUnless(
    robosuite_supported(),
    "robosuite 1.5.2 requires mujoco<=3.3.7; the locomotion stack needs 3.11",
)
class TestMakeEnv(unittest.TestCase):
    """Checks the same task constructs and steps on every arm."""

    def test_rejects_unknown_arm(self):
        with self.assertRaises(ValueError):
            make_env("Lift", "Optimus")

    def test_lift_runs_on_every_arm_with_shared_action_width(self):
        widths = {}
        for arm in embodiments.all_arm_names():
            env = make_env("Lift", arm, horizon=10)
            try:
                env.reset()
                low, high = env.action_spec
                self.assertTrue(np.all(high >= low))
                for _ in range(3):
                    _, reward, _, _ = env.step(np.zeros_like(low))
                self.assertTrue(np.isfinite(reward))
                widths[arm] = action_dim(env)
            finally:
                env.close()

        self.assertEqual(
            len(set(widths.values())),
            1,
            f"delta end-effector control should give one action width, got {widths}",
        )

    def test_joint_count_differs_while_action_width_matches(self):
        panda = make_env("Lift", "Panda", horizon=10)
        ur5e = make_env("Lift", "UR5e", horizon=10)
        try:
            self.assertNotEqual(arm_dof(panda), arm_dof(ur5e))
            self.assertEqual(action_dim(panda), action_dim(ur5e))
        finally:
            panda.close()
            ur5e.close()


if __name__ == "__main__":
    unittest.main()

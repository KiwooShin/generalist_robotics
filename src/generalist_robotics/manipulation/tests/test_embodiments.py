"""Unit tests for the embodiment registry."""

import unittest

from generalist_robotics.manipulation import embodiments, robosuite_supported


@unittest.skipUnless(
    robosuite_supported(),
    "robosuite 1.5.2 requires mujoco<=3.3.7; the locomotion stack needs 3.11",
)
class TestEmbodimentRegistry(unittest.TestCase):
    """Checks the arm registry is internally consistent."""

    def test_all_arm_names_covers_every_split(self):
        names = embodiments.all_arm_names()
        self.assertEqual(len(names), len(set(names)))
        expected = (
            embodiments.TRAINING_ARMS + embodiments.HELDOUT_NEAR_ARMS + embodiments.HELDOUT_FAR_ARMS
        )
        self.assertCountEqual(names, expected)

    def test_arms_in_split_matches_constants(self):
        self.assertEqual(embodiments.arms_in_split("train"), embodiments.TRAINING_ARMS)
        self.assertEqual(embodiments.arms_in_split("heldout_near"), embodiments.HELDOUT_NEAR_ARMS)
        self.assertEqual(embodiments.arms_in_split("heldout_far"), embodiments.HELDOUT_FAR_ARMS)

    def test_arms_in_split_rejects_unknown_split(self):
        with self.assertRaises(ValueError):
            embodiments.arms_in_split("validation")

    def test_get_embodiment_returns_matching_record(self):
        record = embodiments.get_embodiment("UR5e")
        self.assertEqual(record.name, "UR5e")
        self.assertEqual(record.split, "heldout_near")
        self.assertEqual(record.dof, 6)

    def test_get_embodiment_rejects_unknown_arm(self):
        with self.assertRaises(ValueError):
            embodiments.get_embodiment("Optimus")

    def test_held_out_arms_are_excluded_from_training(self):
        held_out = embodiments.HELDOUT_NEAR_ARMS + embodiments.HELDOUT_FAR_ARMS
        for arm in held_out:
            self.assertNotIn(arm, embodiments.TRAINING_ARMS)

    def test_task_suite_is_non_empty_and_unique(self):
        self.assertGreater(len(embodiments.TASK_SUITE), 0)
        self.assertEqual(len(embodiments.TASK_SUITE), len(set(embodiments.TASK_SUITE)))


if __name__ == "__main__":
    unittest.main()

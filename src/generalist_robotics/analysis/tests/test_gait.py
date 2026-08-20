"""Tests for analysis.gait."""

import unittest

import numpy as np

from generalist_robotics.analysis import gait

DT = 0.02
PERIOD_STEPS = 20


def square_gait(
    phases: tuple[float, ...], duty: float = 0.5, cycles: int = 8, period: int = PERIOD_STEPS
) -> np.ndarray:
    """Return a (time, leg) contact trace of square waves at the given relative phases."""
    steps = cycles * period
    time = np.arange(steps)[:, None] / period
    offsets = np.asarray(phases)[None, :]
    return (np.mod(time + offsets, 1.0) < duty).astype(float)


class GaitSignatureTest(unittest.TestCase):
    """gait_signature reads the descriptors of a known contact pattern back out."""

    def test_alternating_biped(self):
        signature = gait.gait_signature(square_gait((0.0, 0.5)), DT)
        self.assertAlmostEqual(signature.duty_factors[0], 0.5, places=6)
        self.assertAlmostEqual(signature.duty_factors[1], 0.5, places=6)
        self.assertAlmostEqual(signature.phase_offsets[0], 0.0, places=6)
        self.assertAlmostEqual(signature.phase_offsets[1], 0.5, places=6)
        self.assertAlmostEqual(signature.stride_frequency, 1.0 / (PERIOD_STEPS * DT), places=6)
        self.assertAlmostEqual(signature.contact_sequence_entropy, 1.0, places=6)

    def test_trot_pairs_legs_diagonally(self):
        signature = gait.gait_signature(square_gait((0.0, 0.5, 0.5, 0.0)), DT)
        offsets = np.asarray(signature.phase_offsets)
        np.testing.assert_allclose(offsets, [0.0, 0.5, 0.5, 0.0], atol=1e-6)
        self.assertAlmostEqual(signature.contact_sequence_entropy, 1.0, places=6)

    def test_walking_quadruped_visits_more_patterns_than_a_trot(self):
        walk = gait.gait_signature(square_gait((0.0, 0.25, 0.5, 0.75), duty=0.75), DT)
        trot = gait.gait_signature(square_gait((0.0, 0.5, 0.5, 0.0)), DT)
        self.assertGreater(walk.contact_sequence_entropy, trot.contact_sequence_entropy)

    def test_standing_still_has_no_stride(self):
        signature = gait.gait_signature(np.ones((100, 4)), DT)
        self.assertEqual(signature.stride_frequency, 0.0)
        self.assertEqual(signature.contact_sequence_entropy, 0.0)
        self.assertEqual(signature.phase_offsets, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(signature.duty_factors, (1.0, 1.0, 1.0, 1.0))

    def test_leg_that_never_lands_reports_no_duty_and_no_phase(self):
        contacts = square_gait((0.0, 0.5, 0.0, 0.0))
        contacts[:, 2:] = 0.0
        signature = gait.gait_signature(contacts, DT)
        self.assertEqual(signature.duty_factors[2:], (0.0, 0.0))
        self.assertEqual(signature.phase_offsets[2:], (0.0, 0.0))

    def test_rejects_malformed_traces(self):
        with self.assertRaises(ValueError):
            gait.gait_signature(np.ones(10), DT)
        with self.assertRaises(ValueError):
            gait.gait_signature(np.ones((1, 2)), DT)
        with self.assertRaises(ValueError):
            gait.gait_signature(np.ones((10, 2)), 0.0)


class GaitDistanceTest(unittest.TestCase):
    """gait_distance is zero on a repeat and large across a change of phasing."""

    def test_identical_gaits_are_at_distance_zero(self):
        signature = gait.gait_signature(square_gait((0.0, 0.5, 0.5, 0.0)), DT)
        self.assertAlmostEqual(gait.gait_distance(signature, signature), 0.0, places=9)

    def test_symmetric(self):
        trot = gait.gait_signature(square_gait((0.0, 0.5, 0.5, 0.0)), DT)
        pace = gait.gait_signature(square_gait((0.0, 0.5, 0.0, 0.5)), DT)
        self.assertAlmostEqual(gait.gait_distance(trot, pace), gait.gait_distance(pace, trot))

    def test_trot_and_pace_differ_only_in_phase(self):
        trot = gait.gait_signature(square_gait((0.0, 0.5, 0.5, 0.0)), DT)
        pace = gait.gait_signature(square_gait((0.0, 0.5, 0.0, 0.5)), DT)
        components = gait.gait_distance_components(trot, pace)
        self.assertAlmostEqual(components["duty"], 0.0, places=6)
        self.assertAlmostEqual(components["frequency"], 0.0, places=6)
        # Two of the four legs swapped which pair they swing with, half a cycle each.
        self.assertAlmostEqual(components["phase"], 0.5, places=6)

    def test_a_leg_arriving_shows_up_as_duty(self):
        biped = gait.gait_signature(square_gait((0.0, 0.5, 0.0, 0.0)) * [1, 1, 0, 0], DT)
        tripod = gait.gait_signature(square_gait((0.0, 0.5, 0.25, 0.0)) * [1, 1, 1, 0], DT)
        components = gait.gait_distance_components(biped, tripod)
        self.assertGreater(components["duty"], 0.1)
        self.assertGreater(gait.gait_distance(biped, tripod), 0.0)

    def test_rejects_signatures_of_different_bodies(self):
        biped = gait.gait_signature(square_gait((0.0, 0.5)), DT)
        quadruped = gait.gait_signature(square_gait((0.0, 0.5, 0.5, 0.0)), DT)
        with self.assertRaises(ValueError):
            gait.gait_distance(biped, quadruped)


class DetectBifurcationTest(unittest.TestCase):
    """detect_bifurcation fires on a step that breaks the path's own pattern, and only then."""

    def smooth_path(self) -> tuple[list[gait.GaitSignature], list[float]]:
        """A path whose duty factors drift a little at every step and whose phasing holds."""
        alphas = [0.1 * index for index in range(11)]
        signatures = [
            gait.gait_signature(square_gait((0.0, 0.5, 0.5, 0.0), duty=0.45 + 0.01 * index), DT)
            for index in range(11)
        ]
        return signatures, alphas

    def test_continuous_deformation_reports_nothing(self):
        signatures, alphas = self.smooth_path()
        self.assertEqual(gait.detect_bifurcation(signatures, alphas), [])

    def test_phase_reorganisation_is_reported(self):
        signatures, alphas = self.smooth_path()
        for index in range(6, 11):
            signatures[index] = gait.gait_signature(square_gait((0.0, 0.5, 0.0, 0.5)), DT)
        self.assertEqual(gait.detect_bifurcation(signatures, alphas), [alphas[6]])

    def test_rates_are_measured_per_unit_alpha(self):
        signatures, alphas = self.smooth_path()
        jumps, rates = gait.gait_change_rates(signatures, alphas)
        self.assertEqual(len(jumps), len(alphas) - 1)
        np.testing.assert_allclose(rates, jumps / 0.1, rtol=1e-9)

    def test_rejects_malformed_paths(self):
        signatures, alphas = self.smooth_path()
        with self.assertRaises(ValueError):
            gait.detect_bifurcation(signatures, alphas[:-1])
        with self.assertRaises(ValueError):
            gait.detect_bifurcation(signatures[:1], alphas[:1])
        with self.assertRaises(ValueError):
            gait.detect_bifurcation(signatures, sorted(alphas, reverse=True))


if __name__ == "__main__":
    unittest.main()

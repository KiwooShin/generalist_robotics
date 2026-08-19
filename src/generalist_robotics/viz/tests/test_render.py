"""Unit tests for the morphing rollout and the offscreen renderer."""

import json
import os
import pathlib
import tempfile
import unittest

import mujoco
import numpy as np

from generalist_robotics.morphology.scaling import MorphParams, apply_morphology, interpolate
from generalist_robotics.viz import render

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"

# The real continuation run this repository ships, used to check the log reader against the
# artefact it actually has to read rather than against a fixture that agrees with it by fiat.
SIMILAR_RUN_DIR = pathlib.Path(__file__).resolve().parents[4] / "artifacts" / "continuation_similar"

# A mesh robot on a textured plane: mesh vertices are what a rescaled model has to push back to
# the graphics context, and the plane is what has to stay fixed while everything else grows.
SCENE_XML = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1"
             markrgb="0 0 0" width="60" height="60"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="5 5"/>
    <mesh name="wedge" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="shell" type="mesh" mesh="wedge"/>
      <body name="link" pos="0.1 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom name="rod" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="drive" joint="hinge" kp="8" forcerange="-3 3"/>
  </actuator>
</mujoco>
"""


def build_scene() -> mujoco.MjModel:
    """Compile the little test scene."""
    return mujoco.MjModel.from_xml_string(SCENE_XML)


def offscreen_renderer(model: mujoco.MjModel, width: int, height: int):
    """Build a small offscreen renderer, or None when this machine has no usable context."""
    try:
        return render.OffscreenRenderer(model, width, height, max_geom=200)
    except Exception:  # noqa: BLE001 - any GL failure means the check cannot run here.
        return None


class LoadRunTest(unittest.TestCase):
    """Reading a continuation run's log back into waypoints."""

    def test_reads_the_shipped_similar_run(self):
        header, waypoints = render.load_run(SIMILAR_RUN_DIR)
        self.assertEqual(header["robot"], "berkeley_humanoid")
        self.assertEqual(len(waypoints), 6)
        self.assertEqual([round(w.alpha, 3) for w in waypoints], [0.0, 0.1, 0.25, 0.475, 0.7, 1.0])
        self.assertEqual(waypoints[-1].params.size_scale, 2.0)

    def test_exactly_one_waypoint_was_fine_tuned(self):
        _, waypoints = render.load_run(SIMILAR_RUN_DIR)
        stops = [w for w in waypoints if w.finetune_steps > 0]
        self.assertEqual(len(stops), 1)
        self.assertAlmostEqual(stops[0].alpha, 0.475)
        self.assertEqual(stops[0].finetune_steps, 6_553_600)
        self.assertEqual(waypoints[-1].cumulative_steps, 6_553_600)

    def test_checkpoint_paths_are_rebuilt_from_the_run_directory(self):
        _, waypoints = render.load_run(SIMILAR_RUN_DIR)
        for waypoint in waypoints:
            self.assertEqual(waypoint.checkpoint.parent, SIMILAR_RUN_DIR)
            self.assertTrue(waypoint.checkpoint.name.startswith("waypoint_"))

    def test_rejects_a_log_without_an_accepted_waypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / render.RUN_LOG_NAME).write_text(
                json.dumps({"record": "config", "robot": "berkeley_humanoid"}) + "\n"
            )
            with self.assertRaises(ValueError):
                render.load_run(path)


class PathCoordinateTest(unittest.TestCase):
    """Choosing a policy and a morphology for a point of the path."""

    def waypoints(self):
        """Three waypoints at alpha 0, 0.5 and 1."""
        return [
            render.Waypoint(
                i, alpha, MorphParams(), True, 0, 0, 0.9, None, 0.4, None, pathlib.Path()
            )
            for i, alpha in enumerate((0.0, 0.5, 1.0))
        ]

    def test_takes_the_last_waypoint_at_or_before_alpha(self):
        waypoints = self.waypoints()
        self.assertEqual(render.waypoint_at(waypoints, 0.0), 0)
        self.assertEqual(render.waypoint_at(waypoints, 0.49), 0)
        self.assertEqual(render.waypoint_at(waypoints, 0.5), 1)
        self.assertEqual(render.waypoint_at(waypoints, 1.0), 2)

    def test_alpha_before_the_first_waypoint_still_resolves(self):
        self.assertEqual(render.waypoint_at(self.waypoints(), -0.2), 0)

    def test_relative_morph_composes_to_the_target(self):
        start = MorphParams(1.0, 1.0, 1.0)
        target = interpolate(start, MorphParams(2.0, 8.0, 16.0), 0.37)
        relative = render.relative_morph(start, target)
        self.assertAlmostEqual(relative.size_scale * start.size_scale, target.size_scale)
        self.assertAlmostEqual(relative.mass_scale * start.mass_scale, target.mass_scale)
        self.assertAlmostEqual(relative.torque_scale * start.torque_scale, target.torque_scale)


class RescaleModelTest(unittest.TestCase):
    """Growing one model in place has to agree with apply_morphology's scaled copy."""

    def test_matches_apply_morphology_on_the_same_factors(self):
        params = MorphParams(1.3, 2.2, 3.1)
        expected = apply_morphology(build_scene(), params)
        grown = build_scene()
        render.rescale_model(grown, params)
        np.testing.assert_allclose(grown.geom_size, expected.geom_size)
        np.testing.assert_allclose(grown.body_mass, expected.body_mass)
        np.testing.assert_allclose(grown.dof_damping, expected.dof_damping)
        np.testing.assert_allclose(grown.actuator_gainprm, expected.actuator_gainprm)
        np.testing.assert_allclose(grown.mesh_vert, expected.mesh_vert)

    def test_two_steps_reach_the_same_body_as_one(self):
        one = build_scene()
        render.rescale_model(one, MorphParams(1.44, 2.0, 4.0))
        two = build_scene()
        render.rescale_model(two, MorphParams(1.2, np.sqrt(2.0), 2.0))
        render.rescale_model(two, MorphParams(1.2, np.sqrt(2.0), 2.0))
        np.testing.assert_allclose(two.geom_size, one.geom_size, rtol=1e-9)
        np.testing.assert_allclose(two.mesh_vert, one.mesh_vert, rtol=1e-5)
        np.testing.assert_allclose(two.body_mass, one.body_mass, rtol=1e-9)

    def test_freezing_the_world_geoms_leaves_the_floor_alone(self):
        model = build_scene()
        sizes = np.array(model.geom_size)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        rod = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rod")
        render.rescale_model(model, MorphParams(2.0, 8.0, 16.0))
        render.freeze_world_geoms(model, sizes)
        np.testing.assert_allclose(model.geom_size[floor], sizes[floor])
        np.testing.assert_allclose(model.geom_size[rod], 2.0 * sizes[rod])

    def test_ground_squares_are_set_in_metres_and_survive_a_morph(self):
        model = build_scene()
        render.set_ground_square_size(model, 1.0)
        material = int(
            model.geom_matid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")]
        )
        np.testing.assert_allclose(model.mat_texrepeat[material], 2.0)
        self.assertTrue(bool(model.mat_texuniform[material]))
        render.rescale_model(model, MorphParams(2.0, 8.0, 16.0))
        np.testing.assert_allclose(model.mat_texrepeat[material], 2.0)


class StoryboardTest(unittest.TestCase):
    """The alpha schedule the film is driven by."""

    def beats(self):
        """Hold, grow, hold."""
        return (
            render.Beat("open", 2.0, 0.0, 0.0),
            render.Beat("grow", 4.0, 0.0, 1.0),
            render.Beat("arrive", 2.0, 1.0, 1.0),
        )

    def test_smoothstep_is_clamped_and_symmetric(self):
        self.assertEqual(render.smoothstep(-1.0), 0.0)
        self.assertEqual(render.smoothstep(2.0), 1.0)
        self.assertAlmostEqual(render.smoothstep(0.5), 0.5)
        self.assertAlmostEqual(render.smoothstep(0.25) + render.smoothstep(0.75), 1.0)

    def test_alpha_holds_then_grows_monotonically(self):
        beats = self.beats()
        self.assertEqual(render.storyboard_seconds(beats), 8.0)
        self.assertAlmostEqual(render.alpha_at(beats, 0.0), 0.0)
        self.assertAlmostEqual(render.alpha_at(beats, 1.9), 0.0)
        self.assertAlmostEqual(render.alpha_at(beats, 6.0), 1.0)
        self.assertAlmostEqual(render.alpha_at(beats, 7.9), 1.0)
        samples = [render.alpha_at(beats, 2.0 + 0.1 * step) for step in range(41)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(samples, samples[1:], strict=False)))

    def test_beat_lookup_reports_progress_through_the_beat(self):
        beat, fraction = render.beat_at(self.beats(), 4.0)
        self.assertEqual(beat.name, "grow")
        self.assertAlmostEqual(fraction, 0.5)

    def test_time_past_the_end_stays_on_the_last_beat(self):
        beat, fraction = render.beat_at(self.beats(), 99.0)
        self.assertEqual(beat.name, "arrive")
        self.assertEqual(fraction, 1.0)


class RendererTest(unittest.TestCase):
    """Offscreen rendering of a model that changes size under the graphics context."""

    def test_renders_a_frame_of_the_requested_shape(self):
        model = build_scene()
        renderer = offscreen_renderer(model, 64, 48)
        if renderer is None:
            self.skipTest("no offscreen graphics context available")
        try:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            render.update_camera(camera, render.CameraRig(), data.qpos[:3])
            frame = renderer.render(data, camera)
            self.assertEqual(frame.shape, (48, 64, 3))
            self.assertEqual(frame.dtype, np.uint8)
        finally:
            renderer.close()

    def test_a_morph_between_frames_changes_the_picture(self):
        model = build_scene()
        renderer = offscreen_renderer(model, 96, 72)
        if renderer is None:
            self.skipTest("no offscreen graphics context available")
        try:
            data = mujoco.MjData(model)
            camera = mujoco.MjvCamera()
            mujoco.mj_forward(model, data)
            render.update_camera(camera, render.CameraRig(), data.qpos[:3])
            before = renderer.render(data, camera)
            render.rescale_model(model, MorphParams(2.0, 8.0, 16.0))
            mujoco.mj_forward(model, data)
            after = renderer.render(data, camera)
            self.assertGreater(np.abs(before.astype(int) - after.astype(int)).mean(), 1.0)
        finally:
            renderer.close()

    def test_the_scale_reference_adds_decor_geoms(self):
        model = build_scene()
        renderer = offscreen_renderer(model, 64, 48)
        if renderer is None:
            self.skipTest("no offscreen graphics context available")
        try:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            render.update_camera(camera, render.CameraRig(), data.qpos[:3])
            renderer.render(data, camera)
            bare = renderer.scene.ngeom
            renderer.render(data, camera, render.ScaleReference(marks=(0.5, 1.0)))
            self.assertGreater(renderer.scene.ngeom, bare)
        finally:
            renderer.close()


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the real morphing walk")
class MorphingWalkerTest(unittest.TestCase):
    """A short real walk on the shipped run, which needs Playground, JAX and the checkpoints."""

    def test_walks_and_grows_without_falling(self):
        walker = render.MorphingWalker(SIMILAR_RUN_DIR)
        walker.run_to(0.0, 1.0)
        for step in range(120):
            walker.step(step / 119.0)
        telemetry = walker.telemetry(1.0)
        self.assertTrue(telemetry.upright)
        self.assertAlmostEqual(telemetry.params.size_scale, 2.0, places=6)
        self.assertGreater(telemetry.standing_height, 1.0)
        self.assertEqual(telemetry.cumulative_steps, 6_553_600)


if __name__ == "__main__":
    unittest.main()

"""Rollouts that grow the robot while the policy keeps walking, rendered offscreen to RGB frames."""

import dataclasses
import json
import math
import pathlib
from collections.abc import Callable, Iterator, Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from generalist_robotics.envs.locomotion import make_locomotion_env
from generalist_robotics.evaluation.rollout import froude_number, nominal_leg_length
from generalist_robotics.morphology.scaling import (
    TIME_LENGTH_POWER,
    MorphParams,
    interpolate,
    refresh_derived_constants,
    scale_lengths,
    scale_masses,
    scale_passive_joints,
    scale_torques,
)
from generalist_robotics.training.ppo import load_checkpoint, make_policy

# Layout of a continuation run directory, written by continuation.path.
RUN_LOG_NAME = "run.jsonl"
WAYPOINT_DIR_TEMPLATE = "waypoint_{:03d}"

# The demo is a single scripted walk, not an evaluation, so the two sources of episode
# randomness the training task uses are switched off: the random shoves that stress a
# policy, and the sensor noise. Both are stated on screen.
DEMO_CONFIG_OVERRIDES: dict[str, object] = {
    "push_config.enable": False,
    "noise_config.level": 0.0,
}

# Playground samples the gait clock from U(1.25, 1.5) Hz at reset; the midpoint is used so
# the clip is reproducible. Under dynamic similarity the phase increment per control step,
# 2 * pi * ctrl_dt * gait_frequency, is size invariant, because ctrl_dt grows as sqrt(k)
# while the gait frequency falls as 1/sqrt(k). It is therefore held fixed as the body grows.
GAIT_FREQUENCY = 1.375

# Time constant of the exponential average behind the on-screen speed, in seconds. A walking
# gait swings the base speed by tens of percent within a stride, so the raw value is unreadable.
SPEED_SMOOTHING_SECONDS = 0.5


@dataclasses.dataclass(frozen=True)
class Waypoint:
    """One accepted point of a continuation run, together with the policy that leaves it.

    Attributes:
        index: position along the path, and the name of the checkpoint directory.
        alpha: path coordinate, 0 at the start morphology and 1 at the target.
        params: the morphology at this alpha.
        viable_before: whether the policy arriving here still locomoted acceptably.
        finetune_steps: environment steps spent fine-tuning here, 0 when none were needed.
        cumulative_steps: fine-tune steps spent up to and including this waypoint.
        survived_before: fraction of the episode survived on arrival.
        survived_after: the same after fine-tuning, or None when none was needed.
        speed_before: mean forward speed on arrival, m/s.
        speed_after: the same after fine-tuning, or None when none was needed.
        checkpoint: directory holding the parameters that leave this waypoint.
    """

    index: int
    alpha: float
    params: MorphParams
    viable_before: bool
    finetune_steps: int
    cumulative_steps: int
    survived_before: float
    survived_after: float | None
    speed_before: float
    speed_after: float | None
    checkpoint: pathlib.Path


def read_run_records(run_dir: pathlib.Path) -> list[dict]:
    """Read the JSON lines a continuation run wrote, in order."""
    log = pathlib.Path(run_dir) / RUN_LOG_NAME
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def load_run(run_dir: pathlib.Path) -> tuple[dict, list[Waypoint]]:
    """Return a continuation run's config header and its accepted waypoints.

    Checkpoint paths are rebuilt from run_dir rather than taken from the log, so a run
    directory stays readable after the repository moves.

    Args:
        run_dir: directory holding run.jsonl and the waypoint checkpoints.

    Raises:
        ValueError: if the log carries no config header or no accepted waypoint.
    """
    run_dir = pathlib.Path(run_dir)
    records = read_run_records(run_dir)
    headers = [record for record in records if record.get("record") == "config"]
    if not headers:
        raise ValueError(f"{run_dir / RUN_LOG_NAME} has no config header")

    waypoints = []
    for record in records:
        if record.get("record") != "waypoint" or not record.get("accepted"):
            continue
        after = record.get("stats_after")
        waypoints.append(
            Waypoint(
                index=int(record["index"]),
                alpha=float(record["alpha"]),
                params=MorphParams(**record["params"]),
                viable_before=bool(record["viable_before"]),
                finetune_steps=int(record["finetune_steps"]),
                cumulative_steps=int(record["cumulative_steps"]),
                survived_before=float(record["stats_before"]["survived_fraction"]),
                survived_after=None if after is None else float(after["survived_fraction"]),
                speed_before=float(record["stats_before"]["mean_forward_speed"]),
                speed_after=None if after is None else float(after["mean_forward_speed"]),
                checkpoint=run_dir / WAYPOINT_DIR_TEMPLATE.format(int(record["index"])),
            )
        )
    if not waypoints:
        raise ValueError(f"{run_dir / RUN_LOG_NAME} has no accepted waypoint")
    return headers[0], waypoints


def waypoint_at(waypoints: Sequence[Waypoint], alpha: float) -> int:
    """Return the index into waypoints of the last waypoint at or before alpha."""
    reached = [i for i, waypoint in enumerate(waypoints) if waypoint.alpha <= alpha + 1e-9]
    return reached[-1] if reached else 0


def relative_morph(current: MorphParams, target: MorphParams) -> MorphParams:
    """Return the morph that takes a body already at current to target."""
    return MorphParams(
        size_scale=target.size_scale / current.size_scale,
        mass_scale=target.mass_scale / current.mass_scale,
        torque_scale=target.torque_scale / current.torque_scale,
    )


def rescale_model(model: mujoco.MjModel, params: MorphParams) -> None:
    """Apply a morph to a model in place, in the order morphology.apply_morphology uses.

    apply_morphology returns a scaled deep copy, which is the right thing for building an
    environment and the wrong thing here: a fresh MjModel every control step would cost a
    deep copy per frame and, worse, would invalidate the renderer's GPU copy of the model.
    Growing one model object instead keeps the body MuJoCo integrates and the body the
    renderer draws literally the same object.
    """
    scale_lengths(model, params.size_scale)
    scale_masses(model, params.mass_scale, params.size_scale)
    scale_passive_joints(model, params.mass_scale, params.size_scale)
    scale_torques(model, params.torque_scale, params.size_scale)
    refresh_derived_constants(model)


def set_ground_square_size(model: mujoco.MjModel, metres: float) -> None:
    """Paint the floor with checker squares of a known side, in place.

    The floor material is textured uniformly in world coordinates, so its repeat count fixes a
    grid spacing in metres rather than in body lengths. The morph never touches a material, so
    squares set here stay exactly the same size on the ground at every body size, which is what
    lets the doubling be read straight off the frame.
    """
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        material = int(model.geom_matid[geom])
        if material >= 0:
            model.mat_texrepeat[material] = GROUND_TEXTURE_REPEAT_METRES / metres
            model.mat_texuniform[material] = 1


def freeze_world_geoms(model: mujoco.MjModel, sizes: np.ndarray) -> None:
    """Restore the geometry of the geoms attached to the world body, in place.

    The ground does not grow with the robot, and it is the video's scale reference: a plane's
    third size entry is the spacing of the grid MuJoCo renders it with, so letting the morph
    scale it would grow the floor markings by exactly the factor the robot grows by and make
    the whole point of the shot invisible. Only the geometry is restored; the contact solver
    parameters stay scaled, since those belong to the collision and follow the robot's clock.
    """
    world = model.geom_bodyid == 0
    model.geom_size[world] = sizes[world]


class MjxDataView:
    """MJX-shaped read-only view of the MjData fields a Playground observation reads.

    A Playground environment computes its observation from an mjx.Data, where site frames are
    3x3 matrices and every field is a JAX array; a CPU MjData stores site frames as nine flat
    numbers in NumPy. Re-exposing the handful of arrays the observation touches lets the
    environment's own observation function be reused verbatim on a CPU rollout, which is what
    keeps this module from reimplementing the policy's input.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.qpos = jnp.asarray(data.qpos)
        self.qvel = jnp.asarray(data.qvel)
        self.sensordata = jnp.asarray(data.sensordata)
        self.site_xmat = jnp.asarray(data.site_xmat).reshape(model.nsite, 3, 3)
        self.site_xpos = jnp.asarray(data.site_xpos)
        self.actuator_force = jnp.asarray(data.actuator_force)


@dataclasses.dataclass(frozen=True)
class Telemetry:
    """The state of the walk at one instant, as the HUD reports it.

    Attributes:
        time: simulated seconds since the start of the walk, on the robot's own clock.
        alpha: path coordinate between the start and target morphologies.
        params: the morphology being simulated.
        standing_height: nominal standing base height of this body, metres.
        base_height: live height of the base above the floor, metres.
        speed: smoothed forward speed, m/s.
        froude: speed**2 / (gravity * standing_height), the size-invariant gait speed.
        commanded_speed: forward speed asked of the policy, m/s.
        cumulative_steps: fine-tune environment steps spent up to this point of the path.
        waypoint: index of the waypoint whose policy is driving the robot.
        upright: whether the base is still the right way up, the environment's own fall test.
    """

    time: float
    alpha: float
    params: MorphParams
    standing_height: float
    base_height: float
    speed: float
    froude: float
    commanded_speed: float
    cumulative_steps: int
    waypoint: int
    upright: bool


@dataclasses.dataclass(frozen=True)
class WalkerConfig:
    """Settings of the scripted walk.

    Attributes:
        command_speed: forward joystick command at size 1, m/s. It follows sqrt(size) up the
            path, which is the speed a dynamically similar robot walks the same gait at.
        seed: seed of the observation-noise stream the environment carries. The pose, the
            heading and the gait clock are fixed rather than sampled, so the clip is
            reproducible; the seed is reported on screen for completeness.
        settle_seconds: simulated seconds walked at the start morphology before the frames
            being written begin, so the gait is established when the video opens.
    """

    command_speed: float = 0.5
    seed: int = 0
    settle_seconds: float = 1.5


class MorphingWalker:
    """A robot rescaled a little at every control step while a policy keeps it walking.

    The walk is stepped by CPU MuJoCo rather than MJX, which is what makes a continuous morph
    affordable: MJX holds the model on the device, so a body that changes every control step
    means a new device model every control step, measured here at 0.25-1.1 s of overhead each,
    against 5 ms to rescale an MjModel in place. Stepping on the CPU also removes any gap
    between the body being simulated and the body being drawn, since both are one object.

    The morph is applied at control-step boundaries only. Nothing else about the state is
    touched: the robot is not lifted to match its new legs, so it grows by pressing itself off
    the floor, exactly as the contact solver has it. At the rates used here a control step
    grows the body by well under a tenth of a percent.

    The controller is the environment's, not a copy of it. The observation the policy sees comes
    from the Playground environment's own observation function, fed through MjxDataView, and the
    default pose, action scale and contact sensors are read off the same environment, so nothing
    here can drift away from what the policy was trained against.
    """

    def __init__(
        self,
        run_dir: pathlib.Path,
        config: WalkerConfig | None = None,
        robot: str | None = None,
    ) -> None:
        self.config = config if config is not None else WalkerConfig()
        header, self.waypoints = load_run(run_dir)
        self.robot = robot if robot is not None else str(header["robot"])
        self.start = MorphParams(**header["start"])
        self.end = MorphParams(**header["end"])

        self.env = make_locomotion_env(
            self.robot,
            self.start,
            config_overrides=dict(DEMO_CONFIG_OVERRIDES),
            scale_time=True,
            scale_task=True,
        )
        self.model = self.env.mj_model
        set_ground_square_size(self.model, GROUND_SQUARE_METRES)
        self.world_geom_sizes = np.array(self.model.geom_size)
        self.data = mujoco.MjData(self.model)
        self.base_timestep = float(self.env.sim_dt)
        self.n_substeps = int(self.env.n_substeps)
        self.base_standing_height = nominal_leg_length(self.env)

        self.default_pose = np.asarray(self.env._default_pose)
        self.action_scale = float(self.env._config.action_scale)
        self.contact_sensor_adr = [
            int(self.model.sensor_adr[sensor_id]) for sensor_id in self.env._feet_floor_found_sensor
        ]
        self.policies = [
            jax.jit(make_policy(self.robot, load_checkpoint(waypoint.checkpoint)[0]))
            for waypoint in self.waypoints
        ]

        self.params = self.start
        self.substeps_taken = 0
        self.waypoint = 0
        self.smoothed_speed = 0.0
        self.standing = True
        self.policy_index_override: int | None = None
        self.info: dict = {}
        self.reset()

    @property
    def time(self) -> float:
        """Simulated seconds since reset, measured on the robot's own clock."""
        return float(self.data.time)

    def reset(self) -> None:
        """Put the robot back at the start morphology, upright at the origin, heading along +x."""
        rescale_model(self.model, relative_morph(self.params, self.start))
        freeze_world_geoms(self.model, self.world_geom_sizes)
        self.params = self.start
        self.model.opt.timestep = self.base_timestep

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0:2] = 0.0
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.model, self.data)

        self.substeps_taken = 0
        self.waypoint = 0
        self.smoothed_speed = 0.0
        self.standing = True
        self.info = {
            "rng": jax.random.PRNGKey(self.config.seed),
            "command": jnp.array([self.config.command_speed, 0.0, 0.0]),
            "last_act": jnp.zeros(self.model.nu),
            "last_last_act": jnp.zeros(self.model.nu),
            "phase_dt": jnp.array([2.0 * math.pi * float(self.env.dt) * GAIT_FREQUENCY]),
            "phase": jnp.array([0.0, math.pi]),
            "feet_air_time": jnp.zeros(len(self.contact_sensor_adr)),
        }

    def set_alpha(self, alpha: float) -> None:
        """Rescale the body in place to the morphology at alpha, and put it on its own clock."""
        target = interpolate(self.start, self.end, alpha)
        if target == self.params:
            return
        rescale_model(self.model, relative_morph(self.params, target))
        freeze_world_geoms(self.model, self.world_geom_sizes)
        self.params = target
        self.model.opt.timestep = self.base_timestep * target.size_scale**TIME_LENGTH_POWER

    def foot_contact(self) -> jnp.ndarray:
        """Per-foot floor contact flags, read from the environment's own contact sensors."""
        return jnp.array([self.data.sensordata[adr] > 0 for adr in self.contact_sensor_adr])

    def act(self, alpha: float) -> None:
        """Take one control action: rescale to alpha, observe, and write the motor targets.

        The single MJX-shaped view built here also serves the speed readout and the fall test,
        which is why neither is recomputed per simulator substep.
        """
        self.set_alpha(alpha)
        reached = waypoint_at(self.waypoints, alpha)
        self.waypoint = (
            reached if self.policy_index_override is None else self.policy_index_override
        )
        self.info["command"] = jnp.array(
            [self.commanded_speed(), 0.0, 0.0], dtype=self.info["command"].dtype
        )
        contact = self.foot_contact()
        view = MjxDataView(self.model, self.data)
        self.update_state(view)
        obs = self.env._get_obs(view, self.info, contact)
        action = np.asarray(self.policies[self.waypoint](obs))

        self.data.ctrl[:] = self.default_pose + action * self.action_scale
        self.info["last_last_act"] = self.info["last_act"]
        self.info["last_act"] = jnp.asarray(action)
        phase = self.info["phase"] + self.info["phase_dt"]
        self.info["phase"] = jnp.fmod(phase + math.pi, 2.0 * math.pi) - math.pi
        self.info["feet_air_time"] = (self.info["feet_air_time"] + float(self.env.dt)) * ~contact

    def commanded_speed(self) -> float:
        """Forward speed asked of the policy at the current size, m/s."""
        return self.config.command_speed * self.params.size_scale**TIME_LENGTH_POWER

    def step(self, alpha: float) -> float:
        """Advance one simulator substep, taking a fresh control action on a control boundary.

        Args:
            alpha: path coordinate the body should be at for this control step.

        Returns:
            The simulated time after the substep, in seconds.
        """
        if self.substeps_taken % self.n_substeps == 0:
            self.act(alpha)
        mujoco.mj_step(self.model, self.data)
        self.substeps_taken += 1
        return self.time

    def update_state(self, view: MjxDataView) -> None:
        """Fold this control step's speed into the average the HUD shows, and test for a fall."""
        speed = abs(float(np.asarray(self.env.get_local_linvel(view))[0]))
        control_period = float(self.model.opt.timestep) * self.n_substeps
        weight = min(1.0, control_period / SPEED_SMOOTHING_SECONDS)
        self.smoothed_speed += weight * (speed - self.smoothed_speed)
        self.standing = bool(np.asarray(self.env.get_gravity(view))[-1] > 0.0)

    def run_to(self, alpha: float, seconds: float) -> None:
        """Walk at a fixed alpha for a stretch of simulated time, rendering nothing."""
        deadline = self.time + seconds
        while self.time < deadline:
            self.step(alpha)

    def telemetry(self, alpha: float) -> Telemetry:
        """Report the current state of the walk."""
        standing_height = self.base_standing_height * self.params.size_scale / self.start.size_scale
        return Telemetry(
            time=self.time,
            alpha=alpha,
            params=self.params,
            standing_height=standing_height,
            base_height=float(self.data.qpos[2]),
            speed=self.smoothed_speed,
            froude=froude_number(self.smoothed_speed, standing_height),
            commanded_speed=self.commanded_speed(),
            cumulative_steps=self.waypoints[self.waypoint].cumulative_steps,
            waypoint=self.waypoint,
            upright=self.standing,
        )


@dataclasses.dataclass(frozen=True)
class CameraRig:
    """A camera that follows the robot along the track without ever changing its distance.

    A fixed distance is the whole reason the growth is visible: the model's own tracking
    camera sits at a body-relative offset, which the morph scales along with every other
    length, so a robot filmed by it stays exactly the same size on screen no matter how large
    it becomes.

    Attributes:
        distance: metres from the look-at point to the camera, held constant.
        azimuth: heading of the view direction in degrees.
        elevation: pitch of the view direction in degrees, negative looking down.
        look_height: height of the look-at point above the floor, metres, held constant so the
            robot rises in frame as it grows.
        lateral_follow: fraction of the robot's sideways drift the camera tracks.

    The default azimuth looks along +y and slightly back along -x, which puts the camera on the
    robot's near side, sends its walk across the frame from left to right, and leaves the far
    side of the track clear for the measuring posts.
    """

    distance: float = 4.3
    azimuth: float = 118.0
    elevation: float = -13.0
    look_height: float = 0.70
    lateral_follow: float = 0.7


@dataclasses.dataclass(frozen=True)
class ScaleReference:
    """A row of fixed-size measuring posts the growing robot is compared against.

    Each post is a plain banded column carrying a highlighted plate at each marked height. The
    posts never scale, so the robot's hips starting level with the lower plate and finishing
    level with the upper one is a direct read of the size doubling.

    Attributes:
        spacing: metres between posts along the track.
        lateral_offset: metres to the far side of the robot, so the posts never occlude it and
            never loom over it in perspective.
        height: total post height, metres.
        band: height of one colour band, metres.
        marks: heights in metres given a highlighted plate.
        thickness: half-width of a post, metres.
    """

    spacing: float = 3.5
    lateral_offset: float = 2.4
    height: float = 1.2
    band: float = 0.3
    marks: tuple[float, ...] = ()
    thickness: float = 0.035


# Restrained, high-contrast decor colours: near-white posts banded with slate, and the two
# marked heights in the same vermilion and teal the HUD uses.
POST_LIGHT_RGBA = (0.93, 0.93, 0.91, 1.0)
POST_DARK_RGBA = (0.42, 0.45, 0.50, 1.0)
MARK_RGBA = ((0.78, 0.25, 0.16, 1.0), (0.06, 0.44, 0.45, 1.0))
# Side of the grid squares the floor is painted with, metres. The squares are the video's
# quantitative scale reference, so the number is chosen to be read off the frame, not inherited.
GROUND_SQUARE_METRES = 1.0

# Length in model units that one repeat of a uniformly mapped 2D texture covers. Measured on
# this scene against decor boxes of known size: at texrepeat 1 the floor's grid cell is 2.0 m
# across, and at texrepeat 2 it is exactly 1.0 m. The floor texture draws one bordered cell per
# repeat, so a repeat is a grid square.
GROUND_TEXTURE_REPEAT_METRES = 2.0


def add_box(
    scene: mujoco.MjvScene,
    position: Sequence[float],
    half_size: Sequence[float],
    rgba: Sequence[float],
) -> bool:
    """Append one axis-aligned decorative box to a scene, if it still has room."""
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(half_size, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1
    return True


def add_scale_reference(
    scene: mujoco.MjvScene, reference: ScaleReference, first_x: float, last_x: float
) -> None:
    """Append the measuring posts covering a stretch of track."""
    first = math.ceil(first_x / reference.spacing) * reference.spacing
    bands = max(1, int(round(reference.height / reference.band)))
    y = reference.lateral_offset
    for step in range(int((last_x - first) / reference.spacing) + 1):
        x = first + step * reference.spacing
        for band in range(bands):
            rgba = POST_LIGHT_RGBA if band % 2 == 0 else POST_DARK_RGBA
            centre = (band + 0.5) * reference.band
            add_box(scene, (x, y, centre), (reference.thickness,) * 2 + (reference.band / 2,), rgba)
        for index, mark in enumerate(reference.marks):
            add_box(
                scene,
                (x, y, mark),
                (reference.thickness * 2.4, reference.thickness * 2.4, 0.012),
                MARK_RGBA[index % len(MARK_RGBA)],
            )


def update_camera(camera: mujoco.MjvCamera, rig: CameraRig, base_position: np.ndarray) -> None:
    """Point a free camera at the robot from the rig's fixed distance and angle."""
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[0] = float(base_position[0])
    camera.lookat[1] = float(base_position[1]) * rig.lateral_follow
    camera.lookat[2] = rig.look_height
    camera.distance = rig.distance
    camera.azimuth = rig.azimuth
    camera.elevation = rig.elevation


class OffscreenRenderer:
    """Offscreen renderer that keeps the GPU's copy of the meshes in step with a morphing model.

    MuJoCo uploads mesh vertices to the graphics context once, when the context is built, so a
    model whose mesh_vert array is rescaled afterwards renders with body frames at the new size
    and mesh surfaces at the old one: the robot comes apart into disconnected pieces. Every
    frame therefore re-uploads the meshes whose scale has moved since the last upload.
    """

    def __init__(
        self, model: mujoco.MjModel, width: int = 1920, height: int = 1080, max_geom: int = 4000
    ) -> None:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.model = model
        self.width = width
        self.height = height
        self.context = mujoco.GLContext(width, height)
        self.context.make_current()
        self.scene = mujoco.MjvScene(model, max_geom)
        self.scene_option = mujoco.MjvOption()
        self.render_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.render_context)
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.buffer = np.empty((height, width, 3), dtype=np.uint8)
        self.uploaded_extent = float(model.stat.extent)

    def refresh_meshes(self) -> None:
        """Re-upload every mesh, which is how a rescaled body reaches the graphics context."""
        for mesh in range(self.model.nmesh):
            mujoco.mjr_uploadMesh(self.model, self.render_context, mesh)
        self.uploaded_extent = float(self.model.stat.extent)

    def render(
        self,
        data: mujoco.MjData,
        camera: mujoco.MjvCamera,
        reference: ScaleReference | None = None,
    ) -> np.ndarray:
        """Draw one frame and return it as an (height, width, 3) uint8 array."""
        self.context.make_current()
        if not math.isclose(float(self.model.stat.extent), self.uploaded_extent, rel_tol=1e-9):
            self.refresh_meshes()
        mujoco.mjv_updateScene(
            self.model,
            data,
            self.scene_option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        if reference is not None:
            span = 1.3 * camera.distance * self.viewport.width / self.viewport.height
            add_scale_reference(
                self.scene, reference, camera.lookat[0] - span, camera.lookat[0] + span
            )
        mujoco.mjr_render(self.viewport, self.scene, self.render_context)
        mujoco.mjr_readPixels(self.buffer, None, self.viewport, self.render_context)
        return np.flipud(self.buffer).copy()

    def close(self) -> None:
        """Release the graphics context."""
        self.render_context.free()
        self.context.free()


@dataclasses.dataclass(frozen=True)
class Beat:
    """One stretch of the storyboard: either a hold at one alpha or a growth between two.

    Attributes:
        name: identifier the HUD keys its captions off.
        seconds: duration in simulated seconds.
        start_alpha: path coordinate at the beginning of the beat.
        end_alpha: path coordinate at the end of the beat.
    """

    name: str
    seconds: float
    start_alpha: float
    end_alpha: float


def smoothstep(fraction: float) -> float:
    """Ease a 0..1 fraction so growth starts and stops without a visible jerk."""
    clamped = min(1.0, max(0.0, fraction))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def beat_at(beats: Sequence[Beat], time: float) -> tuple[Beat, float]:
    """Return the beat covering a time and the 0..1 fraction of the way through it."""
    elapsed = 0.0
    for beat in beats:
        if time < elapsed + beat.seconds or beat is beats[-1]:
            return beat, min(1.0, (time - elapsed) / beat.seconds) if beat.seconds > 0 else 1.0
        elapsed += beat.seconds
    raise ValueError("beats must not be empty")


def alpha_at(beats: Sequence[Beat], time: float) -> float:
    """Return the path coordinate the storyboard asks for at a time."""
    beat, fraction = beat_at(beats, time)
    return beat.start_alpha + (beat.end_alpha - beat.start_alpha) * smoothstep(fraction)


def storyboard_seconds(beats: Sequence[Beat]) -> float:
    """Total simulated duration of a storyboard."""
    return float(sum(beat.seconds for beat in beats))


def walk_frames(
    walker: MorphingWalker,
    renderer: OffscreenRenderer,
    beats: Sequence[Beat],
    fps: int = 60,
    rig: CameraRig | None = None,
    reference: ScaleReference | None = None,
    policy_override: Callable[[Beat, float], int | None] | None = None,
) -> Iterator[tuple[np.ndarray, Telemetry, Beat, float]]:
    """Walk the storyboard, yielding one rendered frame per video frame in real time.

    Frames are emitted on a wall-clock grid rather than one per control step: the control
    period grows as sqrt(size) along the path, so a frame per control step would silently
    slow the clip down as the robot grew. The simulator is advanced substep by substep until
    it passes each frame's timestamp, which lands every frame within one substep, at most
    2.8 ms, of real time.

    Args:
        walker: the robot being grown.
        renderer: offscreen renderer bound to the walker's model.
        beats: the storyboard to walk.
        fps: frames per second of the output, interpreted as real time.
        rig: camera placement, or None for the default.
        reference: measuring posts to draw, or None for a bare floor.
        policy_override: given the beat and how far through it the clip is, the waypoint whose
            policy should drive the robot, or None to follow the path coordinate. It is what
            keeps the arriving policy in charge while the fine-tuning beat plays out.

    Yields:
        The frame, the telemetry behind it, the beat it belongs to, and the fraction of the
        way through that beat.
    """
    rig = rig if rig is not None else CameraRig()
    camera = mujoco.MjvCamera()
    total = storyboard_seconds(beats)
    start = walker.time
    frame = 0
    while frame / fps <= total:
        deadline = start + frame / fps
        beat, fraction = beat_at(beats, walker.time - start)
        if policy_override is not None:
            walker.policy_index_override = policy_override(beat, fraction)
        while walker.time < deadline:
            walker.step(alpha_at(beats, walker.time - start))
        elapsed = walker.time - start
        beat, fraction = beat_at(beats, elapsed)
        update_camera(camera, rig, walker.data.qpos[:3])
        image = renderer.render(walker.data, camera, reference)
        yield image, walker.telemetry(alpha_at(beats, elapsed)), beat, fraction
        frame += 1

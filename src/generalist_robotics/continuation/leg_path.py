"""Walking a locomotion policy along a path on which whole legs grow onto the body.

This module carries three things that would otherwise live apart - a self-contained MJX
locomotion environment for the procedural multiped, the PPO wiring that trains on it, and
the continuation walk itself - because the multiped is not a MuJoCo Playground robot and
milestone M3 owns only this file on the continuation side. envs.locomotion and
training.ppo are both built around Playground's registry: they look a robot's environment
id, its tuned hyperparameters and its network architecture up by name, and a procedurally
generated body has no entry to look up. The alternative, registering a synthetic robot in
Playground, would mean editing files this milestone must not touch and inheriting a
joystick task whose sensors, keyframes and foot names the generated model does not have.
Everything that is not Playground-specific is reused as it stands: the rollout harness and
its Froude viability bar, brax's PPO through training.ppo's own patches and checkpointing,
the lock machinery of morphology.topology, and the predictor-corrector step control of
continuation.path.
"""

import dataclasses
import functools
import json
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

from generalist_robotics.analysis.gait import GaitSignature, gait_signature
from generalist_robotics.continuation.path import (
    ALPHA_TOLERANCE,
    CHECKPOINT_PREFIX,
    CONFIG_RECORD,
    MAX_STEP_ALPHA,
    PROGRESS_DIRNAME,
    STEP_GROWTH,
    WAYPOINT_RECORD,
    append_record,
    stats_record,
)
from generalist_robotics.evaluation.rollout import (
    Policy,
    RolloutStats,
    evaluate_policy,
    is_viable,
    rotate_by_quat,
)
from generalist_robotics.morphology.multiped import (
    CTRL_TIMESTEP,
    SIM_TIMESTEP,
    LegGrowth,
    MultipedSpec,
    apply_leg_growth,
    build_multiped_model,
    foot_site_name,
    foot_velocity_sensor_name,
    growth_by_leg,
    home_ctrl,
    home_qpos,
    leg_geom_names,
    total_mass,
)
from generalist_robotics.morphology.scaling import MorphParams
from generalist_robotics.runtime.gpu import gpu_lock
from generalist_robotics.training import ppo

# How far above the floor the centre of a foot may be and still count as standing on it,
# as a multiple of that foot's own radius. Read per leg off the model rather than as one
# absolute height, because a half-grown leg carries a half-sized foot and would otherwise
# be counted in stance while still in the air.
STANCE_CLEARANCE = 1.5

# Control steps of a gait trace: the first are thrown away because the reset transient is
# not a gait, and the rest are long enough to hold a dozen strides at any stride frequency
# these bodies walk at.
GAIT_SETTLE_STEPS = 50
GAIT_TRACE_STEPS = 400

# Per-waypoint artefacts written next to a checkpoint, so a renderer can rebuild the body
# and replay the gait from the checkpoint directory alone.
WAYPOINT_MORPHOLOGY_FILENAME = "morphology.json"
WAYPOINT_CONTACTS_FILENAME = "contacts.npy"

# PPO hyperparameters for the multiped. Playground's tuned locomotion configs are keyed by
# environment id and there is no entry for a generated body, so these are the Go1 joystick
# settings with the batch shape cut to a body an order of magnitude cheaper to simulate:
# brax requires batch_size * num_minibatches % num_envs == 0, which 512 * 32 and 4096
# satisfy.
MULTIPED_PPO_CONFIG: dict[str, Any] = {
    "num_timesteps": 30_000_000,
    "num_envs": 4096,
    "batch_size": 512,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "unroll_length": 20,
    "learning_rate": 3.0e-4,
    "entropy_cost": 5.0e-3,
    "discounting": 0.97,
    "gae_lambda": 0.95,
    "clipping_epsilon": 0.2,
    "max_grad_norm": 1.0,
    "reward_scaling": 1.0,
    "normalize_observations": True,
    "num_evals": 6,
    "num_eval_envs": 512,
    "num_resets_per_eval": 0,
}

POLICY_HIDDEN_LAYER_SIZES = (256, 256, 128)
VALUE_HIDDEN_LAYER_SIZES = (256, 256, 128)
OBSERVATION_KEY = "state"

# Fixed part of the observation: body-frame angular velocity, the world up axis in body
# frame, body-frame linear velocity and the commanded speed.
OBSERVATION_PREAMBLE_WIDTH = 10

# Scale applied to joint velocities in the observation, which are an order of magnitude
# larger than every other entry and would otherwise dominate the normaliser's early
# statistics.
JOINT_VELOCITY_OBSERVATION_SCALE = 0.1


def default_multiped_config() -> config_dict.ConfigDict:
    """Task specification of the multiped walker: track a forward speed and stay upright.

    The command is one fixed forward speed rather than Playground's random joystick, for
    two reasons. A gait signature is only meaningful if the robot is asked for the same
    thing at every waypoint, and a policy that must cover a symmetric command range spends
    most of its samples on turning and reversing, which this milestone does not measure.

    Nothing here rewards a gait: there is no swing-height target, no air-time bonus and no
    commanded stride frequency, so nothing states which legs should swing together, how
    long a foot should stay down or how often it should cycle. That is deliberate, because
    the gait is the measurement.

    Two weights had to be set against observed failures rather than picked. The alive bonus
    at 0.5 left diving forward - a whole episode of tracking reward compressed into fifty
    steps - worth more than walking, so it is 1.5, at which standing still already beats
    diving and walking beats standing still. And with no slip penalty at all the biped
    learned to skate: fifty million steps of training produced a policy that held 0.49 m/s
    with both feet permanently on the floor, oscillating over a nine-millimetre range and
    never once leaving it, which has a duty factor of one and no gait to speak of. The
    feet_slip term charges a foot for moving while it carries load, which is a statement
    about friction and not about gait: it says a foot must be picked up to be moved
    forward, and says nothing about when.
    """
    return config_dict.create(
        ctrl_dt=CTRL_TIMESTEP,
        sim_dt=SIM_TIMESTEP,
        episode_length=500,
        action_scale=0.4,
        command_velocity=0.5,
        tracking_sigma=0.25,
        min_torso_height=0.30,
        min_upright=0.5,
        joint_reset_noise=0.05,
        height_reset_noise=0.01,
        velocity_reset_noise=0.1,
        reward_scales=config_dict.create(
            tracking_forward=2.0,
            alive=1.5,
            upright=0.5,
            lateral_velocity=-0.5,
            vertical_velocity=-0.3,
            yaw_rate=-0.3,
            feet_slip=-2.0,
            action_rate=-0.01,
            joint_deviation=-0.05,
        ),
    )


def observation_width(spec: MultipedSpec) -> int:
    """Width of the observation vector a spec's environment emits."""
    return OBSERVATION_PREAMBLE_WIDTH + 3 * 3 * spec.n_legs


class MultipedLocomotion(mjx_env.MjxEnv):
    """Flat-ground forward locomotion for a procedural multiped with partly grown legs.

    The action width is the spec's, not the grown body's, which is the property the whole
    milestone rests on: legs that have not grown yet are present in the model as locked
    stubs, so their observation entries and action channels exist and simply drive
    nothing, and a policy carries across the whole path without ever changing shape.
    """

    def __init__(
        self,
        spec: MultipedSpec,
        growth: Sequence[LegGrowth] = (),
        config: config_dict.ConfigDict | None = None,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config or default_multiped_config(), dict(config_overrides or {}))
        self.spec = spec
        self.growth = tuple(growth)
        if float(self._config.sim_dt) != SIM_TIMESTEP:
            raise ValueError(
                f"sim_dt is fixed at {SIM_TIMESTEP} for the multiped, because the leg locks "
                f"are sized against the compiled model's own timestep; got {self._config.sim_dt}"
            )
        self._model = apply_leg_growth(build_multiped_model(spec), spec, self.growth)
        self._mjx = mjx.put_model(self._model)
        self._home_qpos = jnp.asarray(home_qpos(spec))
        self._home_ctrl = jnp.asarray(home_ctrl(spec))
        self._ctrl_range = jnp.asarray(self._model.actuator_ctrlrange)
        self._foot_sites = np.asarray(
            [self._model.site(foot_site_name(spec, leg)).id for leg in range(spec.n_legs)]
        )
        self._stance_heights = jnp.asarray(
            [
                STANCE_CLEARANCE * float(self._model.geom(leg_geom_names(spec, leg)[2]).size[0])
                for leg in range(spec.n_legs)
            ]
        )
        addresses = np.asarray(
            [
                int(self._model.sensor(foot_velocity_sensor_name(spec, leg)).adr[0])
                for leg in range(spec.n_legs)
            ]
        )
        self._foot_velocity_indices = addresses[:, None] + np.arange(3)[None, :]

    @property
    def xml_path(self) -> str:
        """The multiped is generated in memory, so it has no file on disk."""
        return ""

    @property
    def action_size(self) -> int:
        """One channel per actuated joint of the spec, grown or not."""
        return int(self._model.nu)

    @property
    def mj_model(self):
        """The grown and locked MuJoCo model this environment integrates."""
        return self._model

    @property
    def mjx_model(self):
        """The MJX model built from it."""
        return self._mjx

    def torso_rotation(self, data: mjx.Data) -> jax.Array:
        """Inverse of the torso's world orientation, as a wxyz quaternion."""
        return data.qpos[3:7] * jnp.array([1.0, -1.0, -1.0, -1.0])

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Linear velocity of the torso in its own frame, which is what the reward tracks.

        Named for evaluation.rollout, which prefers an environment's own local-velocity
        accessor over reading the free joint, and takes one argument.
        """
        return rotate_by_quat(self.torso_rotation(data), data.qvel[0:3])

    def up_axis(self, data: mjx.Data) -> jax.Array:
        """The world's up direction expressed in the torso frame; its z entry is uprightness."""
        return rotate_by_quat(self.torso_rotation(data), jnp.array([0.0, 0.0, 1.0]))

    def foot_contacts(self, data: mjx.Data) -> jax.Array:
        """Per leg, whether its foot is standing on the floor.

        Stance is read off the height of the foot's centre rather than off the contact
        list, because a foot's own radius is the natural threshold and it is the one
        quantity that shrinks with the leg: a stub at the hip is never in stance, however
        the solver happens to have numbered its contacts.
        """
        return data.site_xpos[self._foot_sites, 2] < self._stance_heights

    def foot_velocities(self, data: mjx.Data) -> jax.Array:
        """World-frame linear velocity of every foot, one row per leg."""
        return data.sensordata[self._foot_velocity_indices]

    def observation(self, data: mjx.Data, last_action: jax.Array) -> dict[str, jax.Array]:
        """Proprioception plus the command, as the single-key dict brax's networks read."""
        state = jnp.concatenate(
            [
                data.qvel[3:6],
                self.up_axis(data),
                self.get_local_linvel(data),
                data.qpos[7:] - self._home_qpos[7:],
                data.qvel[6:] * JOINT_VELOCITY_OBSERVATION_SCALE,
                last_action,
                jnp.array([self._config.command_velocity]),
            ]
        )
        return {OBSERVATION_KEY: state}

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Stand the robot in its home crouch, jittered enough that it cannot memorise one start."""
        rng, joint_key, height_key, velocity_key = jax.random.split(rng, 4)
        joints = jax.random.uniform(
            joint_key,
            (self._model.nu,),
            minval=-self._config.joint_reset_noise,
            maxval=self._config.joint_reset_noise,
        )
        height = jax.random.uniform(
            height_key,
            (),
            minval=-self._config.height_reset_noise,
            maxval=self._config.height_reset_noise,
        )
        qpos = self._home_qpos.at[7:].add(joints).at[2].add(height)
        qvel = (
            jnp.zeros(self._model.nv)
            .at[0:6]
            .set(
                jax.random.uniform(
                    velocity_key,
                    (6,),
                    minval=-self._config.velocity_reset_noise,
                    maxval=self._config.velocity_reset_noise,
                )
            )
        )
        data = mjx_env.make_data(
            self._model, qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self._mjx.impl.value
        )
        data = mjx.forward(self._mjx, data)
        last_action = jnp.zeros(self._model.nu)
        info = {"rng": rng, "last_action": last_action}
        metrics = {"forward_velocity": jnp.zeros(()), "upright": jnp.zeros(())}
        return mjx_env.State(
            data=data,
            obs=self.observation(data, last_action),
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics=metrics,
            info=info,
        )

    def motor_targets(self, action: jax.Array) -> jax.Array:
        """Position targets the action asks for, as an offset from the home crouch."""
        targets = self._home_ctrl + self._config.action_scale * action
        return jnp.clip(targets, self._ctrl_range[:, 0], self._ctrl_range[:, 1])

    def terminated(self, data: mjx.Data) -> jax.Array:
        """Whether the torso has fallen, tipped over, or the integrator has blown up."""
        fallen = data.qpos[2] < self._config.min_torso_height
        tipped = self.up_axis(data)[2] < self._config.min_upright
        broken = jnp.logical_not(jnp.all(jnp.isfinite(data.qpos)))
        return jnp.where(jnp.logical_or(jnp.logical_or(fallen, tipped), broken), 1.0, 0.0)

    def rewards(
        self, data: mjx.Data, action: jax.Array, last_action: jax.Array
    ) -> dict[str, jax.Array]:
        """The reward terms, before their weights: track the command and stay a walker."""
        velocity = self.get_local_linvel(data)
        error = velocity[0] - self._config.command_velocity
        slip = jnp.sum(self.foot_velocities(data)[:, :2] ** 2, axis=1)
        return {
            "tracking_forward": jnp.exp(-(error**2) / self._config.tracking_sigma),
            "alive": jnp.ones(()),
            "upright": self.up_axis(data)[2],
            "lateral_velocity": velocity[1] ** 2,
            "vertical_velocity": velocity[2] ** 2,
            "yaw_rate": data.qvel[5] ** 2,
            "feet_slip": jnp.sum(jnp.where(self.foot_contacts(data), slip, 0.0)),
            "action_rate": jnp.sum((action - last_action) ** 2),
            "joint_deviation": jnp.sum((data.qpos[7:] - self._home_qpos[7:]) ** 2),
        }

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Advance one control step and score it."""
        data = mjx_env.step(self._mjx, state.data, self.motor_targets(action), self.n_substeps)
        terms = self.rewards(data, action, state.info["last_action"])
        scales = self._config.reward_scales
        total = sum(terms[name] * scales[name] for name in terms) * self.dt
        done = self.terminated(data)
        info = dict(state.info)
        info["last_action"] = action
        # Updated rather than rebuilt: brax's evaluation wrapper adds a "reward" entry of
        # its own, and a step that dropped it would change the scan carry's pytree.
        metrics = dict(state.metrics)
        metrics["forward_velocity"] = self.get_local_linvel(data)[0]
        metrics["upright"] = self.up_axis(data)[2]
        return mjx_env.State(
            data=data,
            obs=self.observation(data, action),
            reward=jnp.where(done > 0.5, 0.0, total),
            done=done,
            metrics=metrics,
            info=info,
        )


def make_multiped_env(
    spec: MultipedSpec,
    growth: Sequence[LegGrowth] = (),
    config_overrides: Mapping[str, Any] | None = None,
) -> MultipedLocomotion:
    """Build the locomotion environment of one spec at one growth state."""
    env = MultipedLocomotion(spec, growth, config_overrides=config_overrides)
    if env.observation_size[OBSERVATION_KEY][-1] != observation_width(spec):
        raise RuntimeError(
            f"observation is {env.observation_size} wide but observation_width says "
            f"{observation_width(spec)}; the two must agree for a policy to cross the path"
        )
    return env


def multiped_network_factory() -> Any:
    """Return the PPO network builder both training and inference must agree on."""
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
        value_hidden_layer_sizes=VALUE_HIDDEN_LAYER_SIZES,
        policy_obs_key=OBSERVATION_KEY,
        value_obs_key=OBSERVATION_KEY,
    )


def multiped_ppo_config(
    spec: MultipedSpec,
    num_timesteps: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """PPO keyword arguments for a multiped, with the episode horizon taken from the task."""
    config = dict(MULTIPED_PPO_CONFIG)
    config["network_factory"] = multiped_network_factory()
    config["episode_length"] = int(default_multiped_config().episode_length)
    if num_timesteps is not None:
        config["num_timesteps"] = int(num_timesteps)
    if overrides:
        config.update(overrides)
    del spec  # The batch shape is the same for every leg count these bodies reach.
    return config


@functools.cache
def multiped_ppo_networks(spec: MultipedSpec) -> ppo_networks.PPONetworks:
    """Build the networks a spec's trained parameters belong to.

    The sizes come from the spec rather than from a constructed environment, because they
    are properties of the superset body and are the same at every growth state along the
    path; that invariance is what lets one parameter set be evaluated on every waypoint.
    """
    return multiped_network_factory()(
        {OBSERVATION_KEY: (observation_width(spec),)},
        3 * spec.n_legs,
        preprocess_observations_fn=running_statistics.normalize,
    )


def make_multiped_policy(spec: MultipedSpec, params: object, deterministic: bool = True) -> Policy:
    """Build a Policy callable from trained parameters, as training.ppo does for Playground."""
    inference = ppo_networks.make_inference_fn(multiped_ppo_networks(spec))(
        params, deterministic=deterministic
    )
    base_key = jax.random.PRNGKey(ppo.POLICY_SAMPLING_SEED)

    def policy(observation: dict[str, jax.Array]) -> jax.Array:
        key = base_key if deterministic else ppo.stochastic_action_key(base_key, observation)
        action, _ = inference(observation, key)
        return action

    return policy


def train_multiped_policy(
    spec: MultipedSpec,
    growth: Sequence[LegGrowth] = (),
    num_timesteps: int | None = None,
    seed: int = 0,
    init_params: object | None = None,
    progress_path: pathlib.Path | None = None,
    ppo_overrides: Mapping[str, Any] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> ppo.TrainingResult:
    """Train or fine-tune a policy on one growth state of a multiped.

    This is training.ppo.train_policy for a body Playground does not know about: it goes
    through the same brax entry point, the same compatibility patch and the same progress
    log, and returns the same TrainingResult, so a checkpoint written here is readable by
    training.ppo.load_checkpoint and a walk is a chain of warm starts exactly as in M2.

    Raises:
        runtime.gpu.GpuBusyError: if another process already holds the GPU lock.
    """
    env = make_multiped_env(spec, growth, config_overrides)
    config = multiped_ppo_config(spec, num_timesteps, ppo_overrides)
    log = ppo.ProgressLog(progress_path)
    ppo.restore_replicated_device_put()

    start = time.perf_counter()
    with gpu_lock():
        _, trained_params, final_metrics = ppo_train.train(
            environment=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            seed=seed,
            restore_params=init_params,
            progress_fn=log.record,
            **config,
        )
        jax.block_until_ready(trained_params)
    elapsed = time.perf_counter() - start

    metrics = ppo.float_metrics(final_metrics)
    initial_reward = log.initial_reward()
    if initial_reward is not None:
        metrics[ppo.INITIAL_EVAL_REWARD_KEY] = initial_reward
    steps = log.last_step() or int(config["num_timesteps"])
    return ppo.TrainingResult(
        params=trained_params,
        metrics=metrics,
        num_timesteps=steps,
        wall_clock_seconds=elapsed,
        steps_per_second=steps / elapsed if elapsed > 0.0 else 0.0,
    )


def rollout_contacts(
    env: MultipedLocomotion, policy: Policy, steps: int, rng: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Roll one episode out, recording which feet are in stance and whether it is still alive."""

    def step_once(carry, _):
        state, alive = carry
        following = env.step(state, policy(state.obs))
        alive = jnp.logical_and(alive, following.done < 0.5)
        return (following, alive), (env.foot_contacts(following.data), alive)

    _, (contacts, alive) = jax.lax.scan(
        step_once, (env.reset(rng), jnp.array(True)), None, length=steps
    )
    return contacts, alive


def contact_trace(
    env: MultipedLocomotion,
    policy: Policy,
    seed: int = 0,
    settle_steps: int = GAIT_SETTLE_STEPS,
    trace_steps: int = GAIT_TRACE_STEPS,
) -> np.ndarray:
    """Return a (time, leg) stance trace of one episode, transient and post-fall steps dropped.

    Args:
        env: the body and growth state to walk on.
        policy: the policy to walk with.
        seed: PRNG seed of the episode.
        settle_steps: control steps discarded at the start, where the robot is recovering
            from its reset rather than walking.
        trace_steps: control steps recorded after that.

    Returns:
        The stance trace, which is empty when the policy fell before the transient ended.
    """
    rollout = jax.jit(functools.partial(rollout_contacts, env, policy, settle_steps + trace_steps))
    contacts, alive = rollout(jax.random.PRNGKey(seed))
    live = np.asarray(alive)[settle_steps:]
    trace = np.asarray(contacts, dtype=float)[settle_steps:]
    return trace[: int(np.argmin(live)) if not live.all() else len(live)]


def waypoint_gait(
    env: MultipedLocomotion, policy: Policy, seed: int = 0
) -> tuple[GaitSignature | None, np.ndarray]:
    """Measure the gait one policy walks with on one body, and return the trace behind it.

    Returns:
        The signature and the stance trace it was read from. The signature is None when
        the policy did not stay up long enough to have a gait, which is itself the honest
        answer at a waypoint the policy failed.
    """
    trace = contact_trace(env, policy, seed=seed)
    if trace.shape[0] < 2:
        return None, trace
    return gait_signature(trace, float(env.dt)), trace


def interpolate_growth(
    start: Sequence[LegGrowth], end: Sequence[LegGrowth], alpha: float, spec: MultipedSpec
) -> tuple[LegGrowth, ...]:
    """Blend two growth states linearly, leg by leg.

    Linear because growth is already the coordinate the mass and the length maps are
    linear in and the lock's own map is logarithmic in stiffness, so a straight sweep of
    alpha moves the mechanism smoothly on both axes at once.

    Raises:
        ValueError: if alpha is not finite or leaves [0, 1].
    """
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be finite and in [0, 1], got {alpha!r}")
    low = growth_by_leg(spec, start)
    high = growth_by_leg(spec, end)
    return tuple(
        LegGrowth(leg, (1.0 - alpha) * low[leg] + alpha * high[leg]) for leg in range(spec.n_legs)
    )


@dataclasses.dataclass(frozen=True)
class LegWaypoint:
    """A waypoint on a path along which legs grow onto the body.

    Attributes:
        alpha: path coordinate, 0 at the start growth state and 1 at the target.
        growth: how grown every leg is here.
        total_mass: mass of the body at this waypoint, in kilograms.
        stats_before: rollout statistics of the incoming policy on this body.
        viable_before: whether the incoming policy already locomoted here.
        stats_after: statistics after the last fine-tune round, or None when none was run.
        finetune_steps: environment steps spent fine-tuning at this waypoint.
        cumulative_steps: environment steps spent over the whole run up to and including
            this waypoint.
        signature: gait of the outgoing policy here, or None when it did not stay up.
        contacts: the stance trace that signature was read from.
    """

    alpha: float
    growth: tuple[LegGrowth, ...]
    total_mass: float
    stats_before: RolloutStats
    viable_before: bool
    stats_after: RolloutStats | None
    finetune_steps: int
    cumulative_steps: int
    signature: GaitSignature | None
    contacts: np.ndarray


@dataclasses.dataclass(frozen=True)
class LegContinuationResult:
    """Outcome of walking a policy from one growth state to another.

    Attributes:
        waypoints: every point the walk visited, in order, rejected ones included.
        reached_target: whether the walk arrived at alpha = 1 with a viable policy.
        total_finetune_steps: environment steps spent over the whole run.
        wall_clock_seconds: seconds spent inside the walk.
        final_policy_params: the policy at the last accepted waypoint.
    """

    waypoints: list[LegWaypoint]
    reached_target: bool
    total_finetune_steps: int
    wall_clock_seconds: float
    final_policy_params: object


@dataclasses.dataclass(frozen=True)
class LegContinuationConfig:
    """Everything one waypoint visit needs that does not change along the path.

    Attributes:
        spec: the superset multiped every point of the path is expressed in.
        finetune_timesteps: environment-step budget of one fine-tune round.
        max_finetune_rounds: fine-tune rounds attempted before a waypoint is rejected.
        num_eval_episodes: episodes per viability evaluation.
        seed: base PRNG seed; evaluation uses it directly so every body is judged on the
            same episodes, and fine-tune round r uses seed + r.
        checkpoint_dir: directory receiving per-waypoint checkpoints, or None.
        ppo_overrides: PPO hyperparameters replacing the defaults.
        config_overrides: task-config entries replacing the defaults.
    """

    spec: MultipedSpec
    finetune_timesteps: int
    max_finetune_rounds: int
    num_eval_episodes: int
    seed: int
    checkpoint_dir: pathlib.Path | None = None
    ppo_overrides: Mapping[str, Any] | None = None
    config_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def leg_waypoint_accepted(waypoint: LegWaypoint) -> bool:
    """Whether the policy locomoted at this waypoint, before or after fine-tuning."""
    if waypoint.viable_before:
        return True
    return waypoint.stats_after is not None and is_viable(waypoint.stats_after)


def evaluate_growth(
    config: LegContinuationConfig, growth: tuple[LegGrowth, ...], policy_params: object
) -> tuple[RolloutStats, GaitSignature | None, np.ndarray]:
    """Roll the policy out on one growth state and report both how it did and how it walked."""
    with gpu_lock():
        env = make_multiped_env(config.spec, growth, config.config_overrides)
        policy = make_multiped_policy(config.spec, policy_params)
        stats = evaluate_policy(
            env, policy, num_episodes=config.num_eval_episodes, seed=config.seed
        )
        signature, contacts = waypoint_gait(env, policy, seed=config.seed)
    return stats, signature, contacts


def finetune_progress_path(
    config: LegContinuationConfig, index: int, round_index: int
) -> pathlib.Path | None:
    """Return the JSONL file one fine-tune round streams its metrics to, if any."""
    if config.checkpoint_dir is None:
        return None
    name = f"{CHECKPOINT_PREFIX}_{index:03d}_round_{round_index}.jsonl"
    return pathlib.Path(config.checkpoint_dir) / PROGRESS_DIRNAME / name


def visit_growth(
    config: LegContinuationConfig,
    index: int,
    alpha: float,
    growth: tuple[LegGrowth, ...],
    policy_params: object,
    cumulative_steps: int,
    correct: bool = True,
) -> tuple[LegWaypoint, object]:
    """Test the policy on one growth state and fine-tune it there until it walks.

    This is the corrector of the predictor-corrector loop; the predictor is the step in
    alpha, which grows the legs a little further.
    """
    stats_before, signature, contacts = evaluate_growth(config, growth, policy_params)
    viable_before = is_viable(stats_before)
    trial_params = policy_params
    stats_after: RolloutStats | None = None
    spent = 0

    rounds = 0 if viable_before or not correct else config.max_finetune_rounds
    for round_index in range(rounds):
        result = train_multiped_policy(
            config.spec,
            growth,
            num_timesteps=config.finetune_timesteps,
            seed=config.seed + round_index,
            init_params=trial_params,
            progress_path=finetune_progress_path(config, index, round_index),
            ppo_overrides=config.ppo_overrides,
            config_overrides=config.config_overrides,
        )
        trial_params = result.params
        spent += result.num_timesteps
        stats_after, signature, contacts = evaluate_growth(config, growth, trial_params)
        if is_viable(stats_after):
            break

    waypoint = LegWaypoint(
        alpha=float(alpha),
        growth=tuple(growth),
        total_mass=multiped_total_mass(config.spec, growth),
        stats_before=stats_before,
        viable_before=viable_before,
        stats_after=stats_after,
        finetune_steps=spent,
        cumulative_steps=cumulative_steps + spent,
        signature=signature,
        contacts=contacts,
    )
    return waypoint, trial_params


@functools.cache
def base_multiped_model(spec: MultipedSpec):
    """The fully grown model of a spec, compiled once and kept."""
    return build_multiped_model(spec)


def multiped_total_mass(spec: MultipedSpec, growth: Sequence[LegGrowth]) -> float:
    """Mass of a spec's body at one growth state, in kilograms."""
    return total_mass(apply_leg_growth(base_multiped_model(spec), spec, tuple(growth)))


def signature_record(signature: GaitSignature | None) -> dict[str, Any] | None:
    """Return a gait signature as a JSON-serialisable document, or None."""
    return None if signature is None else dataclasses.asdict(signature)


def morphology_record(spec: MultipedSpec, waypoint: LegWaypoint) -> dict[str, Any]:
    """Return the body of one waypoint, as a document a renderer can rebuild it from.

    MorphParams is written out even though this path holds it at identity: every other
    continuation artefact in the repo carries one, and a reader should be able to tell
    that the body moved along the leg axis alone rather than guess it from an absence.
    """
    return {
        "alpha": waypoint.alpha,
        "spec": dataclasses.asdict(spec),
        "params": dataclasses.asdict(MorphParams()),
        "growth": [dataclasses.asdict(entry) for entry in waypoint.growth],
        "total_mass": waypoint.total_mass,
        "gait": signature_record(waypoint.signature),
    }


def save_leg_waypoint_checkpoint(
    config: LegContinuationConfig,
    waypoint: LegWaypoint,
    policy_params: object,
    index: int,
    seconds: float,
) -> pathlib.Path | None:
    """Persist the policy at one waypoint, next to the body and the gait it was measured on."""
    if config.checkpoint_dir is None:
        return None
    stats = waypoint.stats_after if waypoint.stats_after is not None else waypoint.stats_before
    metrics = {
        "alpha": waypoint.alpha,
        "total_mass": waypoint.total_mass,
        "cumulative_steps": float(waypoint.cumulative_steps),
        "survived_fraction": stats.survived_fraction,
        "mean_forward_speed": stats.mean_forward_speed,
        "episode_return": stats.episode_return,
    }
    result = ppo.TrainingResult(
        params=policy_params,
        metrics=metrics,
        num_timesteps=waypoint.finetune_steps,
        wall_clock_seconds=seconds,
        steps_per_second=waypoint.finetune_steps / seconds if seconds > 0.0 else 0.0,
    )
    directory = pathlib.Path(config.checkpoint_dir) / f"{CHECKPOINT_PREFIX}_{index:03d}"
    saved = ppo.save_checkpoint(result, directory)
    (saved / WAYPOINT_MORPHOLOGY_FILENAME).write_text(
        json.dumps(morphology_record(config.spec, waypoint), indent=2)
    )
    np.save(saved / WAYPOINT_CONTACTS_FILENAME, waypoint.contacts)
    return saved


def leg_waypoint_record(
    spec: MultipedSpec, waypoint: LegWaypoint, index: int, checkpoint: pathlib.Path | None = None
) -> dict[str, Any]:
    """Return the JSON record of one waypoint: what it measured and what it cost."""
    record = {
        "record": WAYPOINT_RECORD,
        "index": index,
        "viable_before": waypoint.viable_before,
        "accepted": leg_waypoint_accepted(waypoint),
        "finetune_steps": waypoint.finetune_steps,
        "cumulative_steps": waypoint.cumulative_steps,
        "stats_before": stats_record(waypoint.stats_before),
        "stats_after": stats_record(waypoint.stats_after),
        "gait_trace_steps": int(waypoint.contacts.shape[0]),
        "checkpoint": None if checkpoint is None else str(checkpoint),
    }
    record.update(morphology_record(spec, waypoint))
    return record


def leg_config_record(
    config: LegContinuationConfig,
    start_growth: tuple[LegGrowth, ...],
    end_growth: tuple[LegGrowth, ...],
    step_alpha: float,
    min_step_alpha: float,
) -> dict[str, Any]:
    """Return the JSON record heading a run log: how to rebuild every state it visits."""
    return {
        "record": CONFIG_RECORD,
        "spec": dataclasses.asdict(config.spec),
        "start_growth": [dataclasses.asdict(entry) for entry in start_growth],
        "end_growth": [dataclasses.asdict(entry) for entry in end_growth],
        "step_alpha": step_alpha,
        "min_step_alpha": min_step_alpha,
        "finetune_timesteps": config.finetune_timesteps,
        "max_finetune_rounds": config.max_finetune_rounds,
        "num_eval_episodes": config.num_eval_episodes,
        "seed": config.seed,
        "config_overrides": dict(config.config_overrides),
        "ppo_overrides": dict(config.ppo_overrides) if config.ppo_overrides else None,
        "checkpoint_dir": (None if config.checkpoint_dir is None else str(config.checkpoint_dir)),
    }


def record_leg_waypoint(
    config: LegContinuationConfig,
    log: pathlib.Path | None,
    waypoint: LegWaypoint,
    index: int,
    policy_params: object,
    seconds: float,
) -> None:
    """Checkpoint the policy at a waypoint and append the waypoint to the run log."""
    checkpoint = save_leg_waypoint_checkpoint(config, waypoint, policy_params, index, seconds)
    if log is not None:
        append_record(log, leg_waypoint_record(config.spec, waypoint, index, checkpoint))


def walk_leg_path(
    spec: MultipedSpec,
    start_growth: tuple[LegGrowth, ...],
    end_growth: tuple[LegGrowth, ...],
    init_policy_params: object,
    step_alpha: float = 0.1,
    min_step_alpha: float = 0.025,
    finetune_timesteps: int = 3_000_000,
    max_finetune_rounds: int = 3,
    num_eval_episodes: int = 8,
    seed: int = 0,
    log_path: pathlib.Path | None = None,
    *,
    checkpoint_dir: pathlib.Path | None = None,
    step_growth: float = STEP_GROWTH,
    max_step_alpha: float = MAX_STEP_ALPHA,
    ppo_overrides: Mapping[str, Any] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> LegContinuationResult:
    """Walk a policy along a path on which legs grow onto the body.

    This is continuation.dof_path.walk_dof_path with a mass axis added and the morphology
    axis dropped. The path still runs in a single model, so the policy's action width
    never changes; what changes is how much of the body those channels are attached to. A
    leg shrunk to a stub at its hip and held rigid is absent, and annealing its growth up
    brings the limb, its mass and its control authority in together, so a policy trained
    on two legs is carried onto four without ever facing a discontinuity in the model.

    Whether it faces one in the *gait* is the question the milestone asks, and the answer
    is measured rather than assumed: every waypoint records the stance trace of the
    outgoing policy, and analysis.gait.detect_bifurcation reads the sequence afterwards.

    The step control is deliberately the one walk_dof_path uses: the predictor takes a
    step of step_alpha in alpha; the corrector asks evaluation.rollout.is_viable whether
    the policy still locomotes and fine-tunes it there for up to max_finetune_rounds
    rounds if not; a waypoint reached for free grows the stride by step_growth up to
    max_step_alpha; a waypoint that needed training holds it; and a rejected waypoint
    halves it and the walk retreats to the last accepted state, stopping when halving
    would go below min_step_alpha.

    Args:
        spec: the superset multiped that hosts the whole path.
        start_growth: how grown each leg is at alpha = 0.
        end_growth: how grown each leg is at alpha = 1.
        init_policy_params: brax parameters in the layout training.ppo produces.
        step_alpha: initial stride in path coordinate.
        min_step_alpha: stride below which the walk gives up.
        finetune_timesteps: environment-step budget of one fine-tune round.
        max_finetune_rounds: rounds attempted before a waypoint is rejected.
        num_eval_episodes: episodes per viability evaluation.
        seed: base PRNG seed for evaluation and training.
        log_path: JSONL file receiving a config record and then one record per waypoint,
            written as each waypoint completes.
        checkpoint_dir: directory receiving one checkpoint per waypoint, each carrying its
            own growth state and stance trace, or None to keep the run in memory.
        step_growth: factor the stride grows by after a free waypoint.
        max_step_alpha: ceiling for that growth.
        ppo_overrides: PPO hyperparameters replacing the defaults.
        config_overrides: task-config entries replacing the defaults.

    Returns:
        Every waypoint visited, whether the target was reached, the total environment-step
        cost, the wall clock, and the policy at the last accepted waypoint.

    Raises:
        ValueError: if the stride, the stride floor, the round budget or the fine-tune
            budget is not usable.
    """
    if not 0.0 < step_alpha <= 1.0:
        raise ValueError(f"step_alpha must be in (0, 1], got {step_alpha!r}")
    if not 0.0 < min_step_alpha <= step_alpha:
        raise ValueError(
            f"min_step_alpha must be in (0, step_alpha], got {min_step_alpha!r} "
            f"with step_alpha {step_alpha!r}"
        )
    if step_growth < 1.0:
        raise ValueError(f"step_growth must be at least 1, got {step_growth!r}")
    if max_finetune_rounds < 0:
        raise ValueError(f"max_finetune_rounds must not be negative, got {max_finetune_rounds!r}")
    if finetune_timesteps < 1:
        raise ValueError(f"finetune_timesteps must be positive, got {finetune_timesteps!r}")

    config = LegContinuationConfig(
        spec=spec,
        finetune_timesteps=int(finetune_timesteps),
        max_finetune_rounds=int(max_finetune_rounds),
        num_eval_episodes=int(num_eval_episodes),
        seed=int(seed),
        checkpoint_dir=None if checkpoint_dir is None else pathlib.Path(checkpoint_dir),
        ppo_overrides=ppo_overrides,
        config_overrides=dict(config_overrides or {}),
    )
    log = pathlib.Path(log_path) if log_path is not None else None
    if log is not None:
        append_record(
            log, leg_config_record(config, start_growth, end_growth, step_alpha, min_step_alpha)
        )

    started = time.perf_counter()
    waypoints: list[LegWaypoint] = []
    policy_params = init_policy_params
    cumulative = 0
    alpha = 0.0
    stride = float(step_alpha)
    ceiling = max(float(step_alpha), float(max_step_alpha))
    reached = False

    anchor, _ = visit_growth(
        config,
        0,
        0.0,
        interpolate_growth(start_growth, end_growth, 0.0, spec),
        policy_params,
        cumulative,
        correct=False,
    )
    waypoints.append(anchor)
    record_leg_waypoint(config, log, anchor, 0, policy_params, time.perf_counter() - started)

    while alpha < 1.0 - ALPHA_TOLERANCE:
        candidate = min(1.0, alpha + stride)
        index = len(waypoints)
        visit_started = time.perf_counter()
        waypoint, trial_params = visit_growth(
            config,
            index,
            candidate,
            interpolate_growth(start_growth, end_growth, candidate, spec),
            policy_params,
            cumulative,
        )
        cumulative = waypoint.cumulative_steps
        waypoints.append(waypoint)
        accepted = leg_waypoint_accepted(waypoint)
        record_leg_waypoint(
            config, log, waypoint, index, trial_params, time.perf_counter() - visit_started
        )
        # Every waypoint compiles a fresh model, and the traces of the ones behind us are
        # dead weight on a machine whose GPU memory is its system memory.
        jax.clear_caches()

        if accepted:
            policy_params = trial_params
            alpha = candidate
            if waypoint.finetune_steps == 0:
                stride = min(stride * step_growth, ceiling)
            reached = alpha >= 1.0 - ALPHA_TOLERANCE
        else:
            stride = (candidate - alpha) / 2.0
            if stride < min_step_alpha:
                break

    return LegContinuationResult(
        waypoints=waypoints,
        reached_target=reached,
        total_finetune_steps=cumulative,
        wall_clock_seconds=time.perf_counter() - started,
        final_policy_params=policy_params,
    )


def accepted_gait_path(
    result: LegContinuationResult,
) -> tuple[list[GaitSignature], list[float]]:
    """Return the gait signatures of the accepted waypoints and the alphas they sit at.

    Rejected waypoints and waypoints where the policy never stood up carry no gait, so
    they are dropped rather than filled in: detect_bifurcation compares neighbours, and a
    missing gait is not a gait that changed.
    """
    signatures: list[GaitSignature] = []
    alphas: list[float] = []
    for waypoint in result.waypoints:
        if waypoint.signature is None or not leg_waypoint_accepted(waypoint):
            continue
        if alphas and waypoint.alpha <= alphas[-1]:
            continue
        signatures.append(waypoint.signature)
        alphas.append(waypoint.alpha)
    return signatures, alphas


def save_leg_run_log(
    result: LegContinuationResult, spec: MultipedSpec, path: pathlib.Path
) -> pathlib.Path:
    """Write a finished walk as one JSON document.

    The JSONL streamed during the run stays the primary record: it also carries the run
    configuration and the per-waypoint checkpoint paths, which a result does not hold.

    Args:
        result: the finished walk.
        spec: the superset body the walk ran in.
        path: JSON file to write, creating parent directories.

    Returns:
        The absolute path written.
    """
    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "spec": dataclasses.asdict(spec),
        "reached_target": result.reached_target,
        "total_finetune_steps": result.total_finetune_steps,
        "wall_clock_seconds": result.wall_clock_seconds,
        "num_waypoints": len(result.waypoints),
        "num_accepted_waypoints": sum(leg_waypoint_accepted(w) for w in result.waypoints),
        "waypoints": [
            leg_waypoint_record(spec, waypoint, index)
            for index, waypoint in enumerate(result.waypoints)
        ],
    }
    path.write_text(json.dumps(document, indent=2))
    return path

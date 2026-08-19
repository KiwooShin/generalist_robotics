"""PPO training of locomotion policies, checkpointed so a run can be continued on a new body."""

import dataclasses
import functools
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from flax.training import orbax_utils
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp

from generalist_robotics.envs.locomotion import environment_id, make_locomotion_env
from generalist_robotics.evaluation.rollout import Policy, episode_length_for
from generalist_robotics.morphology.scaling import MorphParams
from generalist_robotics.runtime.gpu import gpu_lock

# Layout of a checkpoint directory: an orbax PyTree of brax's
# (normalizer, policy, value) parameter tuple, plus the run's own bookkeeping.
PARAMS_DIRNAME = "params"
METADATA_FILENAME = "metadata.json"

# Reward of the very first evaluation, which brax runs before any gradient step. It is the
# warm start's own witness: a cold run scores a random policy here, a run restored from a
# checkpoint scores that checkpoint, so the two are directly comparable.
INITIAL_EVAL_REWARD_KEY = "initial_eval/episode_reward"
EVAL_REWARD_KEY = "eval/episode_reward"

# A run small enough for a unit test: a few thousand environment steps on a handful of
# envs and short episodes. brax requires batch_size * num_minibatches % num_envs == 0.
SMOKE_PPO_OVERRIDES: dict[str, Any] = {
    "num_timesteps": 8_192,
    "num_envs": 32,
    "batch_size": 16,
    "num_minibatches": 4,
    "num_updates_per_batch": 1,
    "unroll_length": 8,
    "num_evals": 2,
    "num_eval_envs": 32,
    "num_resets_per_eval": 0,
    "episode_length": 60,
}

# Base key for stochastic policies; see stochastic_action_key for why one fixed key is
# enough to give a sampling policy fresh noise at every observation.
POLICY_SAMPLING_SEED = 0

# brax 0.14.2 replicates its training state with jax.device_put_replicated, which JAX
# 0.10 removed; restore_replicated_device_put puts the documented replacement back.
REPLICATED_DEVICE_PUT_NAME = "device_put_replicated"
MESH_AXIS_NAME = "device"


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    """Outcome of one PPO run: the policy, its metrics, and what it cost.

    Attributes:
        params: brax's (normalizer, policy, value) parameter tuple. It is exactly what
            train_policy accepts as init_params, which is what makes a continuation path
            a chain of runs rather than a chain of fresh starts.
        metrics: final evaluation metrics as plain floats, plus INITIAL_EVAL_REWARD_KEY
            carrying the reward of the pre-training evaluation.
        num_timesteps: environment steps actually taken, read back from brax's progress
            callback rather than from the request, since brax rounds the request up to a
            whole number of training steps.
        wall_clock_seconds: seconds spent inside brax's train call, compilation included.
        steps_per_second: num_timesteps / wall_clock_seconds. Compilation is charged to
            the run, so a short run reports a pessimistic rate and a long one the true
            throughput.
    """

    params: object
    metrics: dict[str, float]
    num_timesteps: int
    wall_clock_seconds: float
    steps_per_second: float


class ProgressLog:
    """Collector for brax's per-evaluation metrics, optionally streamed to a JSONL file.

    brax reports metrics only through a callback, so this is also where the number of
    environment steps a run really took is observed.
    """

    def __init__(self, path: pathlib.Path | None = None) -> None:
        self.path = pathlib.Path(path) if path is not None else None
        self.records: list[dict[str, float]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def record(self, num_steps: int, metrics: Mapping[str, Any]) -> None:
        """Append one evaluation's metrics, as required by brax's progress_fn protocol."""
        row = {"num_steps": float(num_steps), **float_metrics(metrics)}
        self.records.append(row)
        if self.path is not None:
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

    def last_step(self) -> int:
        """Largest step count reported so far, or 0 when nothing was reported."""
        return int(max((row["num_steps"] for row in self.records), default=0.0))

    def initial_reward(self) -> float | None:
        """Episode reward of the evaluation brax runs before the first gradient step."""
        if not self.records:
            return None
        return self.records[0].get(EVAL_REWARD_KEY)


def device_put_replicated(value: Any, devices: Any) -> Any:
    """Place a copy of a pytree on every given device, along a new leading axis.

    This is the documented replacement for the jax.device_put_replicated that JAX 0.10
    removed: stack one copy per device and shard that axis across them. The mesh is built
    with automatic axis types because brax consumes the result with jax.pmap, which
    rejects the explicit-sharding mesh that jax.make_mesh otherwise produces.
    """
    devices = list(devices)
    stacked = jax.tree_util.tree_map(
        lambda leaf: jnp.stack([jnp.asarray(leaf)] * len(devices)), value
    )
    mesh = jax.make_mesh(
        (len(devices),),
        (MESH_AXIS_NAME,),
        devices=devices,
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    return jax.device_put(stacked, jax.sharding.NamedSharding(mesh, jax.P(MESH_AXIS_NAME)))


def restore_replicated_device_put() -> None:
    """Reinstate jax.device_put_replicated for brax, which still calls it.

    brax 0.14.2 replicates its training state across devices through this JAX 0.10 has
    since deleted, so ppo_train.train raises AttributeError before taking a single step.
    Patching the name back is what makes brax's PPO usable on this JAX; it is a no-op on
    a JAX that still ships the function.
    """
    if not hasattr(jax, REPLICATED_DEVICE_PUT_NAME):
        setattr(jax, REPLICATED_DEVICE_PUT_NAME, device_put_replicated)


def float_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep the scalar entries of a metrics mapping, as Python floats."""
    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        array = jnp.asarray(value)
        if array.size == 1 and jnp.issubdtype(array.dtype, jnp.number):
            scalars[str(key)] = float(array.reshape(()))
    return scalars


def network_factory_for(robot: str) -> Any:
    """Return the tuned PPO network builder for a robot, bound to its architecture.

    Playground stores the architecture as a sub-config rather than a callable, and the
    Berkeley Humanoid uses an asymmetric actor-critic whose value network reads
    privileged_state; building the networks by hand would silently drop that.
    """
    config = locomotion_params.brax_ppo_config(environment_id(robot))
    return functools.partial(ppo_networks.make_ppo_networks, **config.network_factory)


def ppo_config(
    robot: str, num_timesteps: int | None = None, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Playground's tuned PPO hyperparameters for a robot, as keyword arguments to train.

    Args:
        robot: short robot name from envs.locomotion.available_robots.
        num_timesteps: environment-step budget, or None for the tuned value.
        overrides: hyperparameters replacing the tuned ones, e.g. SMOKE_PPO_OVERRIDES.
    """
    config = dict(locomotion_params.brax_ppo_config(environment_id(robot)))
    # The tuned entry is an architecture sub-config; brax wants a callable.
    config["network_factory"] = network_factory_for(robot)
    if num_timesteps is not None:
        config["num_timesteps"] = int(num_timesteps)
    if overrides:
        config.update(overrides)
    return config


def normalizes_observations(robot: str) -> bool:
    """Whether the tuned config trains this robot with observation normalisation."""
    return bool(locomotion_params.brax_ppo_config(environment_id(robot)).normalize_observations)


@functools.cache
def locomotion_ppo_networks(robot: str) -> ppo_networks.PPONetworks:
    """Build the PPO networks a robot's trained parameters belong to.

    Observation and action sizes are read off a freshly constructed environment. They are
    morphology invariant — scaling a body changes no sensor count — so the unmorphed robot
    is the right reference for parameters trained on any point of the continuation path.
    """
    env = make_locomotion_env(robot)
    preprocess = (
        running_statistics.normalize
        if normalizes_observations(robot)
        else (lambda observation, processor_params: observation)
    )
    return network_factory_for(robot)(
        env.observation_size, env.action_size, preprocess_observations_fn=preprocess
    )


def stochastic_action_key(base_key: jax.Array, observation: Mapping[str, jax.Array]) -> jax.Array:
    """Derive a sampling key from an observation, since a Policy is handed no key.

    evaluation.rollout calls a policy inside a jitted scan with the observation as its only
    argument, so a stochastic policy cannot carry a key stream: a Python-side counter would
    be traced once and freeze into a constant. Folding the observation's own bits into a
    fixed base key gives fresh noise at every distinct state and keeps a rollout exactly
    reproducible, which is what an evaluation harness needs.
    """
    bits = jnp.uint32(0)
    for leaf in jax.tree_util.tree_leaves(observation):
        values = jnp.asarray(leaf, jnp.float32).reshape(-1)
        bits = bits + jnp.sum(jax.lax.bitcast_convert_type(values, jnp.uint32))
    return jax.random.fold_in(base_key, bits)


def make_policy(robot: str, params: object, deterministic: bool = True) -> Policy:
    """Build a Policy callable (obs dict -> action) from trained parameters.

    Args:
        robot: short robot name the parameters were trained on.
        params: brax's (normalizer, policy, value) tuple, from TrainingResult.params or
            load_checkpoint.
        deterministic: whether to emit the distribution's mode instead of sampling.

    Returns:
        A callable evaluation.rollout.evaluate_policy accepts directly.
    """
    inference = ppo_networks.make_inference_fn(locomotion_ppo_networks(robot))(
        params, deterministic=deterministic
    )
    base_key = jax.random.PRNGKey(POLICY_SAMPLING_SEED)

    def policy(observation: dict[str, jax.Array]) -> jax.Array:
        key = base_key if deterministic else stochastic_action_key(base_key, observation)
        action, _ = inference(observation, key)
        return action

    return policy


def train_policy(
    robot: str = "berkeley_humanoid",
    params: MorphParams | None = None,
    num_timesteps: int | None = None,
    seed: int = 0,
    init_params: object | None = None,
    progress_path: pathlib.Path | None = None,
    ppo_overrides: Mapping[str, Any] | None = None,
    **env_kwargs: Any,
) -> TrainingResult:
    """Train (or fine-tune) a locomotion policy on a morphed robot.

    Passing init_params restores the normalizer, policy and value parameters into the new
    run's training state before the first gradient step, so the run starts from the given
    policy instead of a random one. That is the whole continuation mechanism: each body
    along the path is trained from the previous body's weights. The optimizer state is
    deliberately not restored — brax does not checkpoint it, and Adam's moments belong to
    the old body's loss surface anyway.

    Training holds runtime.gpu.gpu_lock for its whole duration, because this machine's GPU
    memory is its system memory and two concurrent JAX processes wedge it.

    Args:
        robot: short robot name from envs.locomotion.available_robots.
        params: morphology factors for the body to train on, or None for the stock robot.
        num_timesteps: environment-step budget, or None for Playground's tuned value.
        seed: PRNG seed for network initialisation and rollouts.
        init_params: parameters to warm start from, in the layout TrainingResult.params
            and load_checkpoint produce.
        progress_path: JSONL file to stream per-evaluation metrics to.
        ppo_overrides: PPO hyperparameters replacing the tuned ones, e.g.
            SMOKE_PPO_OVERRIDES for a test-sized run.
        **env_kwargs: forwarded to envs.locomotion.make_locomotion_env, i.e.
            config_overrides, scale_time and scale_task.

    Returns:
        The trained parameters, the final metrics and the run's measured throughput.

    Raises:
        runtime.gpu.GpuBusyError: if another process already holds the GPU lock.
    """
    env = make_locomotion_env(robot, params, **env_kwargs)
    config = ppo_config(robot, num_timesteps, ppo_overrides)
    if not (ppo_overrides and "episode_length" in ppo_overrides):
        # The env may have been given a shorter horizon through config_overrides, and
        # brax would otherwise keep rolling out the tuned robot's episode length.
        config["episode_length"] = episode_length_for(env)
    log = ProgressLog(progress_path)
    restore_replicated_device_put()

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
        # Training is asynchronous until its outputs are read back.
        jax.block_until_ready(trained_params)
    elapsed = time.perf_counter() - start

    metrics = float_metrics(final_metrics)
    initial_reward = log.initial_reward()
    if initial_reward is not None:
        metrics[INITIAL_EVAL_REWARD_KEY] = initial_reward
    steps = log.last_step() or int(config["num_timesteps"])
    return TrainingResult(
        params=trained_params,
        metrics=metrics,
        num_timesteps=steps,
        wall_clock_seconds=elapsed,
        steps_per_second=steps / elapsed if elapsed > 0.0 else 0.0,
    )


def save_checkpoint(result: TrainingResult, path: pathlib.Path) -> pathlib.Path:
    """Write a run's parameters and bookkeeping to a checkpoint directory.

    Args:
        result: the run to persist.
        path: directory to create; an existing checkpoint at the same path is overwritten.

    Returns:
        The absolute path of the checkpoint directory.
    """
    directory = pathlib.Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(
        directory / PARAMS_DIRNAME,
        result.params,
        force=True,
        save_args=orbax_utils.save_args_from_target(result.params),
    )
    metadata = {
        "metrics": result.metrics,
        "num_timesteps": result.num_timesteps,
        "wall_clock_seconds": result.wall_clock_seconds,
        "steps_per_second": result.steps_per_second,
    }
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    return directory


def load_checkpoint(path: pathlib.Path) -> tuple[object, dict[str, float]]:
    """Read back a checkpoint's parameters and metrics.

    The parameters come back in brax's own layout, so they can be handed straight to
    train_policy as init_params or to make_policy.

    Args:
        path: directory written by save_checkpoint.

    Returns:
        The (normalizer, policy, value) parameters and the metrics of the saved run.

    Raises:
        FileNotFoundError: if path holds no checkpoint.
    """
    directory = pathlib.Path(path).expanduser().resolve()
    params_path = directory / PARAMS_DIRNAME
    if not params_path.exists():
        raise FileNotFoundError(f"no checkpoint parameters at {params_path}")
    params = brax_checkpoint.load(params_path)
    metadata_path = directory / METADATA_FILENAME
    metrics: dict[str, float] = {}
    if metadata_path.exists():
        metrics = dict(json.loads(metadata_path.read_text()).get("metrics", {}))
    return params, metrics

"""Reading a foot-contact trace for the qualitative shape of a gait, and for its jumps."""

import dataclasses
import math

import numpy as np

# Weights of the four terms of gait_distance. They sum to one, so a distance is on a
# 0-to-1 scale whatever the leg count. Phase carries the most because a gait is its
# phasing: a trot and a pace have the same duty factors and the same stride frequency and
# differ only in which legs swing together.
DUTY_WEIGHT = 0.30
PHASE_WEIGHT = 0.40
FREQUENCY_WEIGHT = 0.15
ENTROPY_WEIGHT = 0.15

# Smallest gait distance between neighbouring waypoints that can count as a bifurcation.
# The reference event is a quadruped switching between a trot and a pace, which changes
# nothing but which legs swing together and scores exactly 0.20; the threshold sits a
# third below that so the textbook bifurcation clears it with room, while the gradual
# reshaping a continuation step produces stays an order of magnitude below.
MIN_BIFURCATION_JUMP = 0.15

# How many times the median rate of gait change a step must exceed to count as a jump.
# The median is the path's own baseline drift, so this asks for a step that is out of
# family with the rest of the path rather than merely large.
BIFURCATION_RATE_FACTOR = 3.0

# Floor on a denominator, used where a ratio would otherwise divide by an exactly zero
# frequency or an exactly flat contact trace.
EPSILON = 1.0e-9


@dataclasses.dataclass(frozen=True)
class GaitSignature:
    """Contact-pattern descriptor used to detect qualitative gait change.

    Attributes:
        duty_factors: per leg, the fraction of the trace the foot spent in stance.
        phase_offsets: per leg, the phase of its contact cycle relative to leg 0 at the
            stride frequency, in cycles on [0, 1). A leg that never lifts or never lands
            has no phase and reports 0.0; gait_distance discounts it accordingly.
        stride_frequency: frequency in hertz of the dominant mode of the pooled contact
            signal, i.e. how often a foot completes a stance-swing cycle. 0.0 when no
            foot ever changes state.
        contact_sequence_entropy: Shannon entropy in bits of the distribution over the
            distinct contact patterns the trace visits. A stance that never changes
            scores 0, a biped alternating two feet scores at most 1, and a quadruped
            cycling eight support patterns at most 3.
    """

    duty_factors: tuple[float, ...]
    phase_offsets: tuple[float, ...]
    stride_frequency: float
    contact_sequence_entropy: float

    def __post_init__(self):
        object.__setattr__(self, "duty_factors", tuple(float(v) for v in self.duty_factors))
        object.__setattr__(self, "phase_offsets", tuple(float(v) for v in self.phase_offsets))
        object.__setattr__(self, "stride_frequency", float(self.stride_frequency))
        object.__setattr__(self, "contact_sequence_entropy", float(self.contact_sequence_entropy))
        if len(self.duty_factors) != len(self.phase_offsets):
            raise ValueError(
                "duty_factors and phase_offsets must cover the same legs, got "
                f"{len(self.duty_factors)} and {len(self.phase_offsets)}"
            )

    @property
    def num_legs(self) -> int:
        """How many legs this signature describes."""
        return len(self.duty_factors)


def check_contacts(contacts: np.ndarray) -> np.ndarray:
    """Return a contact trace as a float array of zeros and ones, or raise.

    Raises:
        ValueError: if the trace is not a two-dimensional (time, leg) array with at least
            two samples and one leg.
    """
    array = np.asarray(contacts)
    if array.ndim != 2:
        raise ValueError(f"contacts must be a (time, leg) array, got shape {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"contacts must hold at least two samples of one leg, got {array.shape}")
    return (array > 0.5).astype(float)


def contact_codes(contacts: np.ndarray) -> np.ndarray:
    """Return each time step's support pattern as one integer, legs read as bits."""
    binary = check_contacts(contacts)
    weights = 2 ** np.arange(binary.shape[1])
    return (binary * weights).sum(axis=1).astype(np.int64)


def pattern_entropy(contacts: np.ndarray) -> float:
    """Shannon entropy in bits of the distribution over the support patterns visited."""
    _, counts = np.unique(contact_codes(contacts), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def dominant_bin(spectrum: np.ndarray) -> int:
    """Index of the strongest non-constant frequency of a pooled power spectrum."""
    if spectrum.shape[0] < 2:
        return 0
    return int(np.argmax(spectrum[1:]) + 1)


def gait_signature(contacts: np.ndarray, dt: float) -> GaitSignature:
    """Describe a foot-contact trace by its duty factors, phasing, rate and entropy.

    The stride frequency is read off the pooled power spectrum of the contact signals
    rather than by counting touchdowns, because touchdown counting is defeated by the
    contact chatter a stiff foot produces, and the phases are read at that same frequency
    so that they are phases of one shared cycle rather than of four independent ones.

    Args:
        contacts: (time, leg) array, non-zero where that foot is in stance.
        dt: seconds between successive samples.

    Raises:
        ValueError: if the trace is malformed or dt is not positive.
    """
    binary = check_contacts(contacts)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")

    duty = binary.mean(axis=0)
    centred = binary - duty
    coefficients = np.fft.rfft(centred, axis=0)
    frequencies = np.fft.rfftfreq(binary.shape[0], dt)
    pooled = np.abs(coefficients).sum(axis=1) ** 2
    bin_index = dominant_bin(pooled)
    stride = float(frequencies[bin_index]) if pooled[bin_index] > EPSILON else 0.0

    cycling = np.abs(coefficients[bin_index]) > EPSILON
    reference = np.angle(coefficients[bin_index, 0]) if cycling[0] else 0.0
    phases = (np.angle(coefficients[bin_index]) - reference) / (2.0 * math.pi)
    offsets = np.where(cycling, np.mod(phases, 1.0), 0.0)

    return GaitSignature(
        duty_factors=tuple(duty),
        phase_offsets=tuple(offsets),
        stride_frequency=stride,
        contact_sequence_entropy=pattern_entropy(binary),
    )


def cycling_weight(duty: float) -> float:
    """How much a leg actually cycles, 1 at an even duty factor and 0 at a foot that never
    leaves the floor or never reaches it.

    A leg that does not cycle has no phase, so this is what a phase comparison is weighted
    by; it is the duty factor's own variance, normalised to peak at one.
    """
    return float(4.0 * duty * (1.0 - duty))


def circular_distance(first: float, second: float) -> float:
    """Distance between two phases in cycles, on a 0-to-1 scale, the wrap taken care of.

    Antiphase is the farthest two legs can be from one another, so half a cycle scores 1.
    """
    difference = abs(float(first) - float(second)) % 1.0
    return float(2.0 * min(difference, 1.0 - difference))


def check_comparable(a: GaitSignature, b: GaitSignature) -> int:
    """Return the leg count two signatures share, or raise if they do not share one."""
    if a.num_legs != b.num_legs:
        raise ValueError(
            f"signatures describe different bodies: {a.num_legs} legs and {b.num_legs}"
        )
    return a.num_legs


def gait_distance_components(a: GaitSignature, b: GaitSignature) -> dict[str, float]:
    """Return the four terms of gait_distance separately, each on a 0-to-1 scale.

    Reported alongside the total because they answer different questions: a jump in duty
    alone is a leg starting to carry load, while a jump in phase is the legs reorganising
    into a different gait, and only the second is a bifurcation of the gait itself.
    """
    legs = check_comparable(a, b)
    duty = float(np.mean(np.abs(np.asarray(a.duty_factors) - np.asarray(b.duty_factors))))

    weights = [
        min(cycling_weight(a.duty_factors[leg]), cycling_weight(b.duty_factors[leg]))
        for leg in range(legs)
    ]
    distances = [
        circular_distance(a.phase_offsets[leg], b.phase_offsets[leg]) for leg in range(legs)
    ]
    total_weight = sum(weights)
    phase = (
        float(sum(w * d for w, d in zip(weights, distances, strict=True)) / total_weight)
        if total_weight > EPSILON
        else 0.0
    )

    fastest = max(a.stride_frequency, b.stride_frequency, EPSILON)
    frequency = min(1.0, abs(a.stride_frequency - b.stride_frequency) / fastest)
    entropy = min(1.0, abs(a.contact_sequence_entropy - b.contact_sequence_entropy) / legs)

    return {"duty": duty, "phase": phase, "frequency": frequency, "entropy": entropy}


def gait_distance(a: GaitSignature, b: GaitSignature) -> float:
    """Distance between two gaits on a 0-to-1 scale, zero exactly when they match.

    The four terms are weighted by DUTY_WEIGHT, PHASE_WEIGHT, FREQUENCY_WEIGHT and
    ENTROPY_WEIGHT; see gait_distance_components for what each measures.

    Raises:
        ValueError: if the two signatures describe different numbers of legs.
    """
    components = gait_distance_components(a, b)
    return float(
        DUTY_WEIGHT * components["duty"]
        + PHASE_WEIGHT * components["phase"]
        + FREQUENCY_WEIGHT * components["frequency"]
        + ENTROPY_WEIGHT * components["entropy"]
    )


def check_path(signatures: list[GaitSignature], alphas: list[float]) -> np.ndarray:
    """Return the path coordinates as an array, raising unless they index the signatures.

    Raises:
        ValueError: if the two sequences disagree in length, are too short to hold a step,
            or the coordinates do not increase.
    """
    if len(signatures) != len(alphas):
        raise ValueError(
            f"got {len(signatures)} signatures for {len(alphas)} alphas; they must match"
        )
    if len(signatures) < 2:
        raise ValueError("a bifurcation needs at least two waypoints to sit between")
    coordinates = np.asarray(alphas, dtype=float)
    if not np.all(np.diff(coordinates) > 0.0):
        raise ValueError(f"alphas must strictly increase, got {alphas!r}")
    return coordinates


def gait_change_rates(
    signatures: list[GaitSignature], alphas: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gait distance across each step of a path and that distance per unit alpha.

    Raises:
        ValueError: if the path is malformed; see check_path.
    """
    coordinates = check_path(signatures, alphas)
    jumps = np.asarray(
        [gait_distance(signatures[k - 1], signatures[k]) for k in range(1, len(signatures))]
    )
    return jumps, jumps / np.diff(coordinates)


def detect_bifurcation(signatures: list[GaitSignature], alphas: list[float]) -> list[float]:
    """Alphas where the gait changes qualitatively rather than continuously.

    A continuation path is built so that the body changes a little at every step, so a
    gait that merely deforms produces small, comparable gait distances all along it. A
    bifurcation is a step that breaks that pattern twice over: the gait moves at least
    MIN_BIFURCATION_JUMP in one step, and it moves at least BIFURCATION_RATE_FACTOR times
    faster per unit alpha than the path's own median rate. Both tests are needed - the
    absolute one alone fires on a path that is uniformly coarse, and the relative one
    alone fires on the largest of a set of uniformly tiny steps.

    Args:
        signatures: gait signature measured at each waypoint, in path order.
        alphas: strictly increasing path coordinate of each waypoint.

    Returns:
        The alphas of the waypoints the gait jumped on arrival at, in path order; empty
        when the gait deformed continuously all the way along, which is a real result and
        not a failure.

    Raises:
        ValueError: if the path is malformed; see check_path.
    """
    jumps, rates = gait_change_rates(signatures, alphas)
    baseline = float(np.median(rates))
    return [
        float(alphas[index + 1])
        for index, (jump, rate) in enumerate(zip(jumps, rates, strict=True))
        if jump >= MIN_BIFURCATION_JUMP and rate >= BIFURCATION_RATE_FACTOR * baseline
    ]

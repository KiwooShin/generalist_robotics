"""Robot embodiments and the shared task suite used for cross-embodiment experiments."""

from dataclasses import dataclass
from typing import List

TRAINING_ARMS = ["Panda", "Sawyer", "IIWA", "Kinova3"]
HELDOUT_NEAR_ARMS = ["UR5e"]
HELDOUT_FAR_ARMS = ["Jaco"]

TASK_SUITE = [
    "Lift",
    "Stack",
    "PickPlaceCan",
    "NutAssemblySquare",
    "Door",
    "ToolHang",
]


@dataclass(frozen=True)
class Embodiment:
    """A robot arm participating in the cross-embodiment study.

    Attributes:
        name: robosuite robot identifier, e.g. "Panda".
        split: one of "train", "heldout_near", "heldout_far".
        dof: number of controllable arm joints, excluding the gripper.
    """

    name: str
    split: str
    dof: int


EMBODIMENTS = [
    Embodiment("Panda", "train", 7),
    Embodiment("Sawyer", "train", 7),
    Embodiment("IIWA", "train", 7),
    Embodiment("Kinova3", "train", 7),
    Embodiment("UR5e", "heldout_near", 6),
    Embodiment("Jaco", "heldout_far", 7),
]


def all_arm_names() -> List[str]:
    """Return every arm name in the study, training and held out."""
    return [embodiment.name for embodiment in EMBODIMENTS]


def arms_in_split(split: str) -> List[str]:
    """Return the arm names belonging to one split.

    Args:
        split: "train", "heldout_near", or "heldout_far".
    """
    valid_splits = {"train", "heldout_near", "heldout_far"}
    if split not in valid_splits:
        raise ValueError(f"unknown split {split!r}, expected one of {sorted(valid_splits)}")
    return [e.name for e in EMBODIMENTS if e.split == split]


def get_embodiment(name: str) -> Embodiment:
    """Return the Embodiment record for an arm name."""
    for embodiment in EMBODIMENTS:
        if embodiment.name == name:
            return embodiment
    raise ValueError(f"unknown arm {name!r}, expected one of {all_arm_names()}")

"""Deferred manipulation work: robosuite cross-embodiment suite (see plan.md)."""

import mujoco


def robosuite_supported() -> bool:
    """Return True when the installed MuJoCo still exposes the API robosuite 1.5.2 needs.

    robosuite 1.5.2 reads ``MjData.qM``, removed in MuJoCo 3.11. MJX and MuJoCo
    Playground require 3.11, so the locomotion and manipulation stacks cannot share
    one environment. Manipulation is deferred, so its tests skip when incompatible.
    """
    data = mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco/>"))
    return hasattr(data, "qM")

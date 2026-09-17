"""SUMO/TraCI integration for the ITMS prototype.

When SUMO is installed, use ``simulation.sumo_env`` which provides a
``SumoTrafficEnvironment`` class that mirrors the ``TrafficEnvironment``
interface from ``synthetic_sumo.py`` and runs on the real SUMO GUI.

Usage:
    from simulation.sumo_env import SumoTrafficEnvironment
    env = SumoTrafficEnvironment(scenario="normal", steps=120, gui=True)
    state = env.reset()
    # ... use with DQN controller
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict


def is_sumo_available() -> bool:
    return bool(importlib.util.find_spec("traci") and importlib.util.find_spec("sumolib"))


def setup_instructions() -> str:
    return (
        "SUMO Python bindings found. Use `SumoTrafficEnvironment` from "
        "`simulation.sumo_env` to run the 4-way intersection on the real "
        "SUMO GUI. The network, routes, and config files are in "
        "`src/simulation/sumo_config/`."
    )


def describe_required_files(base_dir: str | Path = "src/simulation/sumo_config") -> Dict[str, str]:
    base = Path(base_dir)
    return {
        "network": str(base / "intersection.net.xml"),
        "routes": str(base / "traffic_routes.rou.xml"),
        "config": str(base / "itms.sumocfg"),
    }

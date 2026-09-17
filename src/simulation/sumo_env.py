"""SUMO/TraCI-based 4-way intersection environment.

Mirrors the ``TrafficEnvironment`` interface from ``synthetic_sumo.py`` so
the existing DQN controller and pipeline work unchanged.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from simulation.synthetic_sumo import LANES, PHASES, SCENARIOS


CONFIG_DIR = Path(__file__).resolve().parent / "sumo_config"


def _find_sumo(gui: bool = True) -> str:
    name = "sumo-gui" if gui else "sumo"
    exe = shutil.which(name)
    if exe:
        return exe
    candidates = [
        Path(os.environ.get("SUMO_HOME", "")) / "bin" / name,
        Path("C:/Program Files (x86)/Eclipse/Sumo/bin") / name,
        Path("C:/Program Files/Eclipse/Sumo/bin") / name,
    ]
    for c in candidates:
        if c.with_suffix(".exe").exists():
            return str(c.with_suffix(".exe"))
        if c.exists():
            return str(c)
    return name  # hope it's on PATH


# ---- Phase-to-signal mapping ------------------------------------------------
# The traffic light at "center" controls 20 links (linkIndex 0-19):
#   0-4  : north approach  (north_0, north_1, north_2 → all destinations)
#   5-9  : east approach   (east_0, east_1, east_2   → all destinations)
#   10-14: south approach  (south_0, south_1, south_2 → all destinations)
#   15-19: west approach   (west_0, west_1, west_2   → all destinations)

PHASE_TO_SIGNAL: Dict[int, str] = {
    0: "GGGGGrrrrrrrrrrrrrrr",   # north green
    1: "rrrrrGGGGGrrrrrrrrrrr",  # east  green
    2: "rrrrrrrrrrGGGGGrrrrrr",  # south green
    3: "rrrrrrrrrrrrrrrGGGGG",   # west  green
}

YELLOW_SIGNAL: Dict[int, str] = {
    0: "yyyyyrrrrrrrrrrrrrrr",
    1: "rrrrryyyyyrrrrrrrrrrrr",
    2: "rrrrrrrrrryyyyyrrrrrr",
    3: "rrrrrrrrrrrrrryyyyy",
}

ALL_RED = "rrrrrrrrrrrrrrrrrrrr"


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


class SumoTrafficEnvironment:
    """4-way intersection simulator backed by SUMO/TraCI.

    Implements the same public interface as ``TrafficEnvironment`` in
    ``synthetic_sumo.py`` so the DQN controller and pipeline can use it
    without modification.
    """

    def __init__(
        self,
        scenario: str = "normal",
        steps: int = 120,
        seed: int = 7,
        interval_seconds: int = 3,
        gui: bool = True,
        sumo_cfg: str | Path | None = None,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {sorted(SCENARIOS)}")
        self.scenario_name = scenario
        self.scenario = SCENARIOS[scenario]
        self.steps = steps
        self.seed = seed
        self.interval_seconds = interval_seconds
        self.gui = gui
        self.gui_delay: int = 500  # ms per simulation step in GUI mode
        self.sumo_cfg = Path(sumo_cfg or CONFIG_DIR / "itms.sumocfg")
        self._rows: List[Dict[str, object]] = []
        self._decision_rows: List[Dict[str, object]] = []
        self._lane_vehicles: Dict[str, set] = {lane: set() for lane in LANES}
        self._emergency_seen_at: Optional[int] = None
        self._emergency_cleared_at: Optional[int] = None
        self._current_phase: int = 0
        self._active_preemption: Optional[int] = None
        self._orig_lane_speeds: Dict[str, float] = {}
        self._rng = np.random.default_rng(seed)
        self._tls_id: str = "center"

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        return len(LANES) * 4

    @property
    def action_size(self) -> int:
        return len(PHASES)

    # ── Environment lifecycle ───────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._stop_sumo()
        self._rows.clear()
        self._decision_rows.clear()
        self._lane_vehicles = {lane: set() for lane in LANES}
        self._emergency_seen_at = None
        self._emergency_cleared_at = None
        self._current_phase = 0
        self._active_preemption = None
        self._orig_lane_speeds.clear()
        self._rng = np.random.default_rng(self.seed)
        self._step_index = 0

        cfg_path = str(self.sumo_cfg.resolve())
        binary = _find_sumo(gui=self.gui)

        sumo_args = [
            binary,
            "-c", cfg_path,
            "--start",
            "--no-step-log",
            "--time-to-teleport", "300",
            "--waiting-time-memory", "1000",
            "--collision.action", "none",
        ]

        import traci
        traci.start(sumo_args, label="itms")

        if self.gui:
            try:
                traci.gui.setSchema("View #0", "real world")
            except Exception:
                pass
        traci.vehicletype.setColor("passenger", (255, 255, 255, 255))
        traci.vehicletype.setColor("emergency", (255, 0, 0, 255))
        traci.vehicletype.setColor("ambulance", (0, 255, 0, 255))

        self._ensure_custom_tl_logic()

        return self.state_vector()

    def step(self, action: int, reason: str = "policy", decision_latency: float = 0.0) -> StepResult:
        import traci

        if self._step_index >= self.steps:
            raise RuntimeError("Environment is already done. Call reset() to run another episode.")
        if action < 0 or action >= len(PHASES):
            raise ValueError(f"Invalid action {action}; expected 0..{len(PHASES) - 1}.")

        # Clamp the DQN action to a valid range
        action = max(0, min(action, len(PHASES) - 1))

        # Inject vehicles for this decision step (interval_seconds of sim time)
        for _ in range(self.interval_seconds):
            self._inject_vehicles()
            self._inject_emergency_if_due()
            self._enforce_preemption(action)
            traci.simulationStep()
            sleep(0.05)

        green_lane = PHASES[self._current_phase]

        if self._active_preemption is not None:
            self._boost_emergency_lane()

        # Clear active preemption once all priority vehicles have passed
        if self._active_preemption is not None:
            if self._emergency_preempt_phase() is None and self._ambulance_preempt_phase() is None:
                self._active_preemption = None
                self._restore_lane_speeds()

        # Read metrics after the step
        queues: Dict[str, int] = {}
        waiting_times: Dict[str, float] = {}
        vehicle_counts: Dict[str, int] = {}
        throughputs: Dict[str, int] = {}

        for lane in LANES:
            lane_id = f"{lane}_0"
            q_len = int(traci.lane.getLastStepHaltingNumber(lane_id))
            queues[lane] = q_len
            total_wait = float(traci.lane.getWaitingTime(lane_id))
            waiting_times[lane] = total_wait / max(1, q_len)
            vehicle_counts[lane] = int(traci.lane.getLastStepVehicleNumber(lane_id))

            current_ids = set(traci.lane.getLastStepVehicleIDs(lane_id))
            prev_ids = self._lane_vehicles[lane]
            throughputs[lane] = len(prev_ids - current_ids)
            self._lane_vehicles[lane] = current_ids

        # Emergency checks
        emergency_pending = {
            lane: 1 if self._has_emergency_on_lane(lane) else 0
            for lane in LANES
        }
        if self._emergency_seen_at is not None and self._emergency_cleared_at is None:
            if all(v == 0 for v in emergency_pending.values()):
                self._emergency_cleared_at = self._step_index

        # Accident simulation: block lane
        for lane in LANES:
            if self._accident_active(lane):
                for idx in range(3):
                    traci.lane.setMaxSpeed(f"{lane}_{idx}", 0.5)

        accident_alert = {
            lane: self._accident_alert(lane) for lane in LANES
        }

        # Reward
        total_throughput = sum(throughputs.values())
        total_queue = sum(queues.values())
        total_wait = sum(waiting_times.values())
        emergency_penalty = sum(emergency_pending.values()) * 12.0
        accident_penalty = sum(1 for v in accident_alert.values() if v) * 4.0
        reward = float(total_throughput * 2.0 - total_queue * 0.25 - emergency_penalty - accident_penalty)

        self._decision_rows.append({
            "timestamp": self._step_index,
            "selected_phase": green_lane,
            "green_duration": self.interval_seconds,
            "reason": reason,
            "decision_latency_seconds": round(decision_latency, 6),
        })

        for lane in LANES:
            self._rows.append({
                "scenario": self.scenario_name,
                "step": self._step_index,
                "lane_id": lane,
                "queue_length": int(queues[lane]),
                "waiting_time": int(waiting_times[lane]),
                "vehicle_count": int(vehicle_counts[lane]),
                "throughput": int(throughputs[lane]),
                "signal_phase": green_lane,
                "emergency_count": int(emergency_pending[lane]),
                "accident_alert": bool(accident_alert[lane]),
            })

        self._step_index += 1
        done = self._step_index >= self.steps

        info = {
            "green_lane": green_lane,
            "throughput": throughputs,
            "emergency_clearance_time": self.emergency_clearance_time(),
            "decision_latency_seconds": decision_latency,
            "accident_alert_lanes": [lane for lane in LANES if accident_alert[lane]],
        }

        return StepResult(self.state_vector(queues, emergency_pending, accident_alert), reward, done, info)

    # ── Metrics access ──────────────────────────────────────────────────────

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def decisions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._decision_rows)

    def emergency_clearance_time(self) -> Optional[int]:
        if self._emergency_seen_at is None or self._emergency_cleared_at is None:
            return None
        return (self._emergency_cleared_at - self._emergency_seen_at + 1) * self.interval_seconds

    # ── Internal helpers ────────────────────────────────────────────────────

    def state_vector(
        self,
        queues: Optional[Dict[str, int]] = None,
        emergencies: Optional[Dict[str, int]] = None,
        accident_flags: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        import traci

        if queues is None:
            queues = {}
            for lane in LANES:
                queues[lane] = int(traci.lane.getLastStepHaltingNumber(f"{lane}_0"))
        if emergencies is None:
            emergencies = {lane: 1 if self._has_emergency_on_lane(lane) else 0 for lane in LANES}
        if accident_flags is None:
            accident_flags = {lane: self._accident_alert(lane) for lane in LANES}

        q = np.array([queues[lane] for lane in LANES], dtype=float) / 30.0
        d = np.array([queues[lane] * self.interval_seconds for lane in LANES], dtype=float) / 90.0
        e = np.array([emergencies[lane] for lane in LANES], dtype=float)
        a = np.array([1.0 if accident_flags[lane] else 0.0 for lane in LANES])

        return np.concatenate([q, d, e, a])

    def _ensure_custom_tl_logic(self) -> None:
        import traci

        try:
            traci.trafficlight.setProgram(self._tls_id, "0")
        except Exception:
            pass

    def _apply_action(self, action: int) -> None:
        import traci

        if action != self._current_phase:
            yellow = YELLOW_SIGNAL.get(self._current_phase, ALL_RED)
            traci.trafficlight.setRedYellowGreenState(self._tls_id, yellow)
            traci.simulationStep()
            self._current_phase = action

        signal = PHASE_TO_SIGNAL[action]
        traci.trafficlight.setRedYellowGreenState(self._tls_id, signal)

    def _inject_vehicles(self) -> None:
        import traci

        rates: Dict[str, float] = dict(self.scenario["arrival_rates"])
        for lane in LANES:
            rate = rates.get(lane, 2.0)
            if self._rng.random() < rate / (self.interval_seconds * 2):
                route = self._random_route(lane)
                try:
                    veh_id = f"v_{self._step_index}_{lane}_{self._rng.integers(1e6)}"
                    traci.vehicle.add(
                        veh_id,
                        route,
                        typeID="passenger",
                        departLane="best",
                    )
                    traci.vehicle.setColor(veh_id, (255, 255, 255, 255))  # white
                except traci.exceptions.TraCIException:
                    pass

    def _random_route(self, lane: str) -> str:
        direction = self._rng.choice(["through", "left", "right"], p=[0.5, 0.25, 0.25])
        if lane == "north":
            return self._rng.choice(["n_to_s", "n_to_e", "n_to_w"])
        elif lane == "south":
            return self._rng.choice(["s_to_n", "s_to_w", "s_to_e"])
        elif lane == "east":
            return self._rng.choice(["e_to_w", "e_to_n", "e_to_s"])
        else:
            return self._rng.choice(["w_to_e", "w_to_s", "w_to_n"])

    def _inject_emergency_if_due(self) -> None:
        import traci

        emergency = self.scenario.get("emergency")
        if not emergency:
            return
        assert isinstance(emergency, dict)

        route_map = {"north": "n_to_s", "east": "e_to_w", "south": "s_to_n", "west": "w_to_e"}

        if "rate" in emergency:
            if self._rng.random() < float(emergency["rate"]):
                lane = str(self._rng.choice(LANES))
                route = route_map.get(lane, "n_to_s")
                is_emergency = self._rng.random() < float(emergency.get("emergency_ratio", 0.5))
                try:
                    if is_emergency:
                        veh_id = f"emergency_{self._step_index}_{lane}_{self._rng.integers(1e6)}"
                        traci.vehicle.add(veh_id, route, typeID="emergency", departLane="best", departSpeed="max")
                        traci.vehicle.setColor(veh_id, (255, 0, 0, 255))
                        if self._emergency_seen_at is None:
                            self._emergency_seen_at = self._step_index
                    else:
                        veh_id = f"ambulance_{self._step_index}_{lane}_{self._rng.integers(1e6)}"
                        traci.vehicle.add(veh_id, route, typeID="ambulance", departLane="best")
                        traci.vehicle.setColor(veh_id, (0, 255, 0, 255))
                except traci.exceptions.TraCIException:
                    pass
            return

        emergency_step = int(emergency["step"])
        if self._step_index == emergency_step and self._emergency_seen_at is None:
            lane = str(emergency["lane"])
            route = route_map.get(lane, "n_to_s")
            try:
                veh_id = f"emergency_{self._step_index}"
                traci.vehicle.add(
                    veh_id,
                    route,
                    typeID="emergency",
                    departLane="best",
                    departSpeed="max",
                )
                traci.vehicle.setColor(veh_id, (255, 0, 0, 255))
                self._emergency_seen_at = self._step_index
            except traci.exceptions.TraCIException:
                pass

    def _has_emergency_on_lane(self, lane: str) -> bool:
        import traci

        try:
            vehicles = traci.lane.getLastStepVehicleIDs(f"{lane}_0")
            for v in vehicles:
                if v.startswith("emergency_"):
                    return True
        except Exception:
            pass
        return False

    def _count_preempt(self, id_prefix: str) -> Dict[str, int]:
        import traci
        counts = {lane: 0 for lane in LANES}
        try:
            for veh_id in traci.vehicle.getIDList():
                if veh_id.startswith(id_prefix):
                    edge = traci.vehicle.getRoadID(veh_id)
                    if edge in counts:
                        counts[edge] += 1
        except Exception:
            pass
        return counts

    def _pick_preempt_phase(self, counts: Dict[str, int], current_phase: Optional[int]) -> Optional[int]:
        lane_to_phase = {"north": 0, "east": 1, "south": 2, "west": 3}
        phase_to_lane = {v: k for k, v in lane_to_phase.items()}

        if current_phase is not None:
            cur_lane = phase_to_lane.get(current_phase)
            if cur_lane and counts.get(cur_lane, 0) > 0:
                return current_phase

        best_lane = max(counts, key=counts.get)
        if counts[best_lane] > 0:
            return lane_to_phase[best_lane]
        return None

    def _emergency_preempt_phase(self) -> Optional[int]:
        return self._pick_preempt_phase(self._count_preempt("emergency_"), self._active_preemption)

    def _ambulance_preempt_phase(self) -> Optional[int]:
        return self._pick_preempt_phase(self._count_preempt("ambulance_"), self._active_preemption)

    def _enforce_preemption(self, dqn_action: int) -> None:
        import traci

        red = self._emergency_preempt_phase()
        if red is not None:
            self._active_preemption = red
            self._current_phase = red
            traci.trafficlight.setRedYellowGreenState(self._tls_id, PHASE_TO_SIGNAL[red])
            return

        green = self._ambulance_preempt_phase()
        if green is not None:
            self._active_preemption = green
            self._current_phase = green
            traci.trafficlight.setRedYellowGreenState(self._tls_id, PHASE_TO_SIGNAL[green])
            return

        if self._active_preemption is not None:
            self._active_preemption = None
            self._restore_lane_speeds()

        if dqn_action != self._current_phase:
            self._current_phase = dqn_action
        traci.trafficlight.setRedYellowGreenState(self._tls_id, PHASE_TO_SIGNAL[dqn_action])

    def _boost_emergency_lane(self) -> None:
        import traci

        lane_to_phase = {"north": 0, "east": 1, "south": 2, "west": 3}
        phase_to_lane = {v: k for k, v in lane_to_phase.items()}
        lane = phase_to_lane.get(self._active_preemption)
        if lane is None:
            return
        for idx in range(3):
            try:
                lid = f"{lane}_{idx}"
                if lid not in self._orig_lane_speeds:
                    self._orig_lane_speeds[lid] = traci.lane.getMaxSpeed(lid)
                traci.lane.setMaxSpeed(lid, 30.0)
            except Exception:
                pass

    def _restore_lane_speeds(self) -> None:
        import traci

        for lid, orig in self._orig_lane_speeds.items():
            try:
                traci.lane.setMaxSpeed(lid, orig)
            except Exception:
                pass
        self._orig_lane_speeds.clear()

    def _accident_active(self, lane: str) -> bool:
        accident = self.scenario.get("accident")
        if not accident:
            return False
        assert isinstance(accident, dict)
        return (
            lane == str(accident["lane"])
            and int(accident["start"]) <= self._step_index < int(accident["end"])
        )

    def _accident_alert(self, lane: str) -> bool:
        import traci

        if not self._accident_active(lane):
            return False
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(f"{lane}_0")
            stuck = sum(
                1 for v in vehicles
                if traci.vehicle.getSpeed(v) < 0.5
            )
            return stuck >= 4
        except Exception:
            return False

    def _stop_sumo(self) -> None:
        try:
            import traci
            traci.close()
        except Exception:
            pass


# ── Convenience wrappers matching synthetic_sumo.py interface ──────────────

def fixed_timer_action(step: int, cycle: int = 8) -> int:
    return (step // cycle) % len(PHASES)


def run_fixed_timer(
    scenario: str,
    steps: int = 120,
    seed: int = 7,
    cycle: int = 8,
    gui: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    env = SumoTrafficEnvironment(scenario=scenario, steps=steps, seed=seed, gui=gui)
    state = env.reset()
    done = False
    while not done:
        t0 = perf_counter()
        action = fixed_timer_action(env._step_index, cycle=cycle)
        latency = perf_counter() - t0
        result = env.step(action, reason="fixed_timer", decision_latency=latency)
        state = result.state
        done = result.done
    return env.metrics_frame(), env.decisions_frame(), _summarize(env, controller="fixed_timer")


PolicyFn = Callable[[np.ndarray, "SumoTrafficEnvironment"], Tuple[int, str]]


def run_policy(
    scenario: str,
    policy: PolicyFn,
    steps: int = 120,
    seed: int = 7,
    gui: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    env = SumoTrafficEnvironment(scenario=scenario, steps=steps, seed=seed, gui=gui)
    state = env.reset()
    done = False
    while not done:
        t0 = perf_counter()
        action, reason = policy(state, env)
        latency = perf_counter() - t0
        result = env.step(action, reason=reason, decision_latency=latency)
        state = result.state
        done = result.done
    return env.metrics_frame(), env.decisions_frame(), _summarize(env, controller="ai_controller")


def _summarize(env: SumoTrafficEnvironment, controller: str = "ai_controller") -> Dict[str, object]:
    metrics = env.metrics_frame()
    decisions = env.decisions_frame()
    return {
        "scenario": env.scenario_name,
        "controller": controller,
        "average_waiting_time": round(float(metrics["waiting_time"].mean()), 3),
        "average_queue_length": round(float(metrics["queue_length"].mean()), 3),
        "total_throughput": int(metrics["throughput"].sum()),
        "max_queue_length": int(metrics["queue_length"].max()),
        "emergency_clearance_time_seconds": env.emergency_clearance_time(),
        "average_decision_latency_seconds": round(float(decisions["decision_latency_seconds"].mean()), 6),
        "latency_under_3_seconds": bool(decisions["decision_latency_seconds"].max() < 3.0),
        "accident_alert_count": int(metrics["accident_alert"].sum()),
    }


def combine_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)

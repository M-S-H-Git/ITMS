from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import time
import numpy as np
import pandas as pd


LANES: Tuple[str, ...] = ("north", "east", "south", "west")
PHASES: Tuple[str, ...] = ("north", "east", "south", "west")

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "normal": {
        "arrival_rates": {"north": 2.0, "east": 1.8, "south": 2.1, "west": 1.7},
    },
    "peak": {
        "arrival_rates": {"north": 6.0, "east": 5.5, "south": 5.8, "west": 5.2},
    },
    "emergency": {
        "arrival_rates": {"north": 2.0, "east": 1.8, "south": 2.1, "west": 1.7},
        "emergency": {"rate": 0.03, "emergency_ratio": 0.5},
    },
    "accident": {
        "arrival_rates": {"north": 2.0, "east": 1.8, "south": 2.1, "west": 1.7},
        "accident": {"lane": "east", "start": 30, "end": 80},
    },
}


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


class TrafficEnvironment:
    def __init__(
        self,
        scenario: str = "normal",
        steps: int = 120,
        seed: int = 7,
        interval_seconds: int = 3,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {sorted(SCENARIOS)}")
        self.scenario_name = scenario
        self.scenario_cfg = SCENARIOS[scenario]
        self.steps = steps
        self.seed = seed
        self.interval_seconds = interval_seconds
        self._rng = np.random.default_rng(seed)
        self._step_index = 0
        self._current_phase = 0
        self._rows: List[Dict[str, object]] = []
        self._decision_rows: List[Dict[str, object]] = []
        self.queues: Dict[str, int] = {lane: 0 for lane in LANES}
        self.emergency_pending: Dict[str, bool] = {lane: False for lane in LANES}
        self.accident_active: Dict[str, bool] = {lane: False for lane in LANES}
        self.accident_stopped: Dict[str, int] = {lane: 0 for lane in LANES}
        self._emergency_seen_at: Optional[int] = None
        self._emergency_cleared_at: Optional[int] = None

    @property
    def state_size(self) -> int:
        return len(LANES) * 4

    @property
    def action_size(self) -> int:
        return len(PHASES)

    def reset(self) -> np.ndarray:
        self._step_index = 0
        self._current_phase = 0
        self.queues = {lane: 0 for lane in LANES}
        self.emergency_pending = {lane: False for lane in LANES}
        self.accident_active = {lane: False for lane in LANES}
        self.accident_stopped = {lane: 0 for lane in LANES}
        self._rows.clear()
        self._decision_rows.clear()
        self._emergency_seen_at = None
        self._emergency_cleared_at = None
        self._rng = np.random.default_rng(self.seed)
        return self.state_vector()

    def step(self, action: int, reason: str = "policy", decision_latency: float = 0.0) -> StepResult:
        if self._step_index >= self.steps:
            raise RuntimeError("Environment is already done. Call reset() to run another episode.")

        green_lane = PHASES[action]
        self._current_phase = action

        self._sample_arrivals()
        self._sample_emergency()

        accident_alert: Dict[str, bool] = {}
        for lane in LANES:
            accident = self.scenario_cfg.get("accident")
            if accident and lane == accident["lane"]:
                active = accident["start"] <= self._step_index < accident["end"]
            else:
                active = False
            self.accident_active[lane] = active

            if active:
                if self.queues[lane] > 0:
                    self.accident_stopped[lane] += 1
                else:
                    self.accident_stopped[lane] = 0
                alert = self.accident_stopped[lane] >= 4
            else:
                self.accident_stopped[lane] = 0
                alert = False
            accident_alert[lane] = alert

        capacity = 0.0 if self.accident_active.get(green_lane, False) else 2.0
        moved = min(self.queues[green_lane], capacity)
        self.queues[green_lane] -= moved

        if self.emergency_pending[green_lane] and moved >= 1:
            self.emergency_pending[green_lane] = False
            if self._emergency_cleared_at is None:
                self._emergency_cleared_at = self._step_index

        throughputs = {lane: 0 for lane in LANES}
        throughputs[green_lane] = int(moved)

        total_throughput = sum(throughputs.values())
        total_queue = sum(self.queues.values())
        emergency_penalty = sum(1 for v in self.emergency_pending.values() if v) * 12.0
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
                "queue_length": int(self.queues[lane]),
                "waiting_time": int(self.queues[lane] * self.interval_seconds),
                "vehicle_count": int(self.queues[lane]),
                "throughput": int(throughputs[lane]),
                "signal_phase": green_lane,
                "emergency_count": 1 if self.emergency_pending[lane] else 0,
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

        return StepResult(self.state_vector(), reward, done, info)

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def decisions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._decision_rows)

    def emergency_clearance_time(self) -> Optional[int]:
        if self._emergency_seen_at is None or self._emergency_cleared_at is None:
            return None
        return (self._emergency_cleared_at - self._emergency_seen_at + 1) * self.interval_seconds

    def state_vector(
        self,
        queues: Optional[Dict[str, int]] = None,
        emergencies: Optional[Dict[str, int]] = None,
        accident_flags: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        if queues is None:
            queues = self.queues
        if emergencies is None:
            emergencies = {lane: (1 if self.emergency_pending[lane] else 0) for lane in LANES}
        if accident_flags is None:
            accident_flags = {lane: self.accident_active.get(lane, False) for lane in LANES}

        q = np.array([queues[lane] for lane in LANES], dtype=float) / 30.0
        d = np.array([queues[lane] * self.interval_seconds for lane in LANES], dtype=float) / 90.0
        e = np.array([emergencies[lane] for lane in LANES], dtype=float)
        a = np.array([1.0 if accident_flags[lane] else 0.0 for lane in LANES])

        return np.concatenate([q, d, e, a])

    def _sample_arrivals(self) -> None:
        rates: Dict[str, float] = dict(self.scenario_cfg.get("arrival_rates", {}))
        for lane in LANES:
            rate = rates.get(lane, 2.0)
            arrivals = int(self._rng.poisson(rate / self.interval_seconds))
            self.queues[lane] += arrivals

    def _sample_emergency(self) -> None:
        emergency = self.scenario_cfg.get("emergency")
        if not emergency:
            return
        if "rate" in emergency:
            if self._rng.random() < float(emergency["rate"]):
                lane = str(self._rng.choice(LANES))
                self.emergency_pending[lane] = True
                if self._emergency_seen_at is None:
                    self._emergency_seen_at = self._step_index
            return
        step_cfg = int(emergency["step"])
        if self._step_index == step_cfg and self._emergency_seen_at is None:
            lane = str(emergency["lane"])
            self.emergency_pending[lane] = True
            self._emergency_seen_at = self._step_index


PolicyFn = Callable[[np.ndarray, "TrafficEnvironment"], Tuple[int, str]]


def fixed_timer_action(step: int, cycle: int = 8) -> int:
    return (step // cycle) % len(PHASES)


def run_fixed_timer(
    scenario: str,
    steps: int = 120,
    seed: int = 7,
    cycle: int = 8,
    gui: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    env = TrafficEnvironment(scenario=scenario, steps=steps, seed=seed)
    state = env.reset()
    done = False
    while not done:
        t0 = time.perf_counter()
        action = fixed_timer_action(env._step_index, cycle=cycle)
        latency = time.perf_counter() - t0
        result = env.step(action, reason="fixed_timer", decision_latency=latency)
        state = result.state
        done = result.done
    return env.metrics_frame(), env.decisions_frame(), _summarize(env, controller="fixed_timer")


def run_policy(
    scenario: str,
    policy: PolicyFn,
    steps: int = 120,
    seed: int = 7,
    gui: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    env = TrafficEnvironment(scenario=scenario, steps=steps, seed=seed)
    state = env.reset()
    done = False
    while not done:
        t0 = time.perf_counter()
        action, reason = policy(state, env)
        latency = time.perf_counter() - t0
        result = env.step(action, reason=reason, decision_latency=latency)
        state = result.state
        done = result.done
    return env.metrics_frame(), env.decisions_frame(), _summarize(env, controller="ai_controller")


def _summarize(env: TrafficEnvironment, controller: str = "ai_controller") -> Dict[str, object]:
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

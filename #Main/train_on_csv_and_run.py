from __future__ import annotations
import sys
import time
import webbrowser
from time import sleep
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from control.dqn_controller import DQNController, controller_policy
from simulation.sumo_env import SumoTrafficEnvironment, _summarize
from simulation.synthetic_sumo import LANES, PHASES
from dashboard import write_dashboard_html

from traci.exceptions import FatalTraCIError

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CSV_PATH = DATA_DIR / "fixed_timer_metrics.csv"


def load_states_and_actions(csv_path: str | Path) -> tuple[list[np.ndarray], list[int], int, int]:
    """Load CSV and extract (state, action) pairs for behavior cloning."""
    df = pd.read_csv(csv_path)
    lane_to_idx = {lane: i for i, lane in enumerate(LANES)}
    states, actions = [], []

    for scenario in df["scenario"].unique():
        sc_df = df[df["scenario"] == scenario]
        for step in sc_df["step"].unique():
            step_data = sc_df[sc_df["step"] == step]
            if len(step_data) != len(LANES):
                continue

            state = np.zeros(len(LANES) * 4, dtype=float)
            for _, row in step_data.iterrows():
                li = lane_to_idx[row["lane_id"]]
                state[li] = float(row["queue_length"]) / 30.0
                state[len(LANES) + li] = float(row["waiting_time"]) / 90.0
                state[2 * len(LANES) + li] = float(row["emergency_count"])
                state[3 * len(LANES) + li] = 1.0 if row["accident_alert"] else 0.0

            signal = str(step_data.iloc[0]["signal_phase"])
            action = PHASES.index(signal) if signal in PHASES else 0
            states.append(state)
            actions.append(action)

    return states, actions, len(LANES) * 4, len(PHASES)


def train_supervised(controller: DQNController, states: list[np.ndarray], actions: list[int]) -> None:
    """Train DQN via behavior cloning: minimize MSE between Q-values and one-hot action targets."""
    rng = np.random.default_rng(42)
    indices = list(range(len(states)))
    for epoch in range(50):
        rng.shuffle(indices)
        total_loss = 0.0
        batch_size = 64
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            for idx in batch_idx:
                state = states[idx]
                target = controller.q_values(state).copy()
                target[actions[idx]] = 1.0
                controller._train_single(state, target)
                total_loss += float(np.mean((controller.q_values(state) - target) ** 2))
        avg_loss = total_loss / max(len(indices) // batch_size, 1)
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}")


def main() -> None:
    print("Loading fixed_timer_metrics.csv...")
    states, actions, state_size, action_size = load_states_and_actions(CSV_PATH)
    print(f"  Loaded {len(states)} state-action pairs, state_size={state_size}, action_size={action_size}")

    controller = DQNController(state_size, action_size, seed=17, learning_rate=0.005)

    print("Training DQN via behavior cloning on logged data...")
    train_supervised(controller, states, actions)

    controller.epsilon = 0.0
    policy = controller_policy(controller)

    print("\nLaunching SUMO GUI — runs indefinitely (close window to stop)")
    print("  Vehicles: WHITE normal, RED emergency")
    print("  Controls: Space=pause, F3=faster, F4=slower")
    print("  Step info printed every 10 steps\n")

    env = SumoTrafficEnvironment(scenario="emergency", steps=999999, gui=True, seed=42)
    state = env.reset()
    step_no = 0
    try:
        while True:
            try:
                t0 = time.perf_counter()
                action, reason = policy(state, env)
                latency = time.perf_counter() - t0
                result = env.step(action, reason=reason, decision_latency=latency)
                state = result.state
                step_no += 1
                if step_no % 10 == 0:
                    rw = result.reward
                    gl = result.info["green_lane"]
                    print(f"  Step {step_no:6d}  phase={gl:6s}  reward={rw:.1f}")
            except (FatalTraCIError, ConnectionError, OSError):
                print("\nSUMO GUI closed by user.")
                break
            except Exception as exc:
                print(f"\nSimulation error ({exc}). Restarting...")
                env._stop_sumo()
                sleep(0.5)
                env = SumoTrafficEnvironment(scenario="emergency", steps=999999, gui=True, seed=42)
                state = env.reset()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            env._stop_sumo()
        except Exception:
            pass

    print(f"Ran {step_no} steps. Done.")

    # ── Generate dashboard ────────────────────────────────────────────────
    print("Generating dashboard...")
    summary_dict = _summarize(env, controller="ai_controller")
    summary = pd.DataFrame([summary_dict])

    metrics = env.metrics_frame()
    decisions = env.decisions_frame()

    lane_stats = metrics.groupby("lane_id").agg(
        predicted_queue_length=("queue_length", "mean")
    ).reset_index()

    def congestion_level(q):
        if q > 150:
            return "high"
        elif q > 80:
            return "medium"
        else:
            return "low"

    lane_stats["congestion_level"] = lane_stats["predicted_queue_length"].apply(congestion_level)
    lane_stats["scenario"] = env.scenario_name
    predictions = lane_stats[["scenario", "lane_id", "predicted_queue_length", "congestion_level"]]

    DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "reports" / "dashboard.html"
    write_dashboard_html(summary, predictions, decisions, metrics, DASHBOARD_PATH)
    webbrowser.open(str(DASHBOARD_PATH.resolve()))
    print(f"Dashboard opened at {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
